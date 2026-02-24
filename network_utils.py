import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import args_config
from torchvision import datasets, transforms
import gc
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from statistics import mean
import math
from training_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
cudnn.deterministic = True

args = args_config.get_args()


"""
QFConvBN2dLIF — Conv + BN (folded into conv) + LIF
most optimized network block for inference
"""
class QFConvBN2dLIF(nn.Module):
    def __init__(self, conv_module, bn_module, lif_module, num_bits_w=4, num_bits_bias=4, num_bits_u=4):
        super(QFConvBN2dLIF,self).__init__()
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.lif_module = lif_module
        self.num_bits_w = num_bits_w
        self.num_bits_bias = num_bits_bias
        self.num_bits_u = num_bits_u

        # Initialize shared scaling factor beta (= alpha in MINT paper, Section IV-C)
        # Formula: 2 * mean(|W|) / sqrt(2^(n-1) - 1):
        # sets an initial quantization range proportional to the weight magnitude.
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()


# Fold BatchNorm into Conv weights and bias for inference
    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std   
            weight = self.conv_module.weight * gamma_.reshape(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            # print(std.shape)
            # print(self.conv_module.weight.shape)
            weight = self.conv_module.weight * gamma_.reshape(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
        return weight, bias


    def forward(self, x):
        # During training, update BN stats
        # during inference, use running stats
        if self.training:  
            # Compute mean/var for BN from current batch
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW (change tensor dimension order. N=batch size, C=channels, H=height, W=width)
            y = y.reshape(self.conv_module.out_channels, -1) # CNHW -> (C,NHW)
            mean = y.mean(1)
            var = y.var(1)
            self.bn_module.running_mean = \
                self.bn_module.momentum * self.bn_module.running_mean + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = \
                self.bn_module.momentum * self.bn_module.running_var + \
                (1 - self.bn_module.momentum) * var
        else:
            # Use running mean/var for inference
            mean = self.bn_module.running_mean
            var = self.bn_module.running_var
        std = torch.sqrt(var + self.bn_module.eps)
        weight, bias = self.fold_bn(mean, std)

        # print("w max:", weight.max())
        # print("b max:", bias.max())
        # print("w min:", weight.min())
        # print("b min:", bias.min())
        # if self.scaling is None:
        #     
        #     self.scaling = nn.ParameterList([nn.Parameter(torch.tensor([alpha])) for i in range(1)]).cuda()
        # else:
        #     qweight = w_q(weight, self.num_bits_w, self.scaling[0])
        
        # Quantize weights and bias if enabled
        if args.wq:
            if args.share:
                qweight,beta = w_q(weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(weight, self.num_bits_w)
        else:
            qweight = weight
        if args.bq:
            if args.share:
                qbias,beta = w_q(bias, self.num_bits_bias, beta)
            else:
                qbias = b_q(bias, self.num_bits_bias)
        else:
            qbias = bias

        # Apply quantized Conv2d
        x = F.conv2d(x, qweight,qbias,
                        stride=self.conv_module.stride,
                        padding=self.conv_module.padding,
                        dilation=self.conv_module.dilation,
                        groups=self.conv_module.groups)
        
        # get spikes from lif neuron with or without shared scaling factor beta, depending on args.share
        if args.share:
            s = self.lif_module(x, args.share, beta)
        else:
            s = self.lif_module(x, args.share, 0)
        return s


"""
QConv2dLIF — Quantized Conv + LIF (no BatchNorm)
Used for layers that don't have BN, e.g. the first layer.
Simpler than QFConvBN2dLIF: no BN folding needed.
"""
class QConv2dLIF(nn.Module):
    def __init__(self, conv_module, lif_module, num_bits_w=4, num_bits_u=4):
        super(QConv2dLIF,self).__init__()

        self.conv_module = conv_module
        self.lif_module = lif_module

        self.num_bits_w = num_bits_w
        self.num_bits_u = num_bits_u
        
        # Initialize shared scaling factor beta (= alpha in MINT paper, Section IV-C)
        # Formula: 2 * mean(|W|) / sqrt(2^(n-1) - 1):
        # sets an initial quantization range proportional to the weight magnitude.
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()

    def forward(self, x):
        # Quantize weights
        if args.wq:
            if args.share:
                qweight,beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight

        # Convolution with (possibly quantized) weights; bias is NOT quantized here
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
                                        padding=self.conv_module.padding,
                                        dilation=self.conv_module.dilation,
                                        groups=self.conv_module.groups)

        # get spikes from lif neuron with or without shared scaling factor beta, depending on args.share
        if args.share:
            s = self.lif_module(x, args.share, beta, bias=0)
        else:
            s = self.lif_module(x, args.share, 0, bias=0)
        # else:
        #     if args.wq:
        #         if args.share:
        #             qweight,beta = w_q_inference(self.conv_module.weight, self.num_bits_w, self.beta[0])
        #         else:
        #             qweight = b_q_inference(self.conv_module.weight, self.num_bits_w)
        #     else:
        #         qweight = self.conv_module.weight
        #     # print(torch.unique(qweight).shape)
        #     # qweight= w_q(self.weight, self.num_bits_weight, in_alpha)
        #     x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
        #                                     padding=self.conv_module.padding,
        #                                     dilation=self.conv_module.dilation,
        #                                     groups=self.conv_module.groups)

        #     if args.share:
        #         s = self.lif_module(x, args.share, beta, bias=0)
        #     else:
        #         s = self.lif_module(x, args.share, 0, bias=0)
        
        #return spikes from LIF neuron
        return s


"""
QConvBN2dLIF — Quantized Conv + BN + LIF
Like QConv2dLIF but with a BN applied AFTER the conv (not folded).
Note: BN is applied at full precision here — no BN folding like in QFConvBN2dLIF.
"""
class QConvBN2dLIF(nn.Module):
    def __init__(self, conv_module, bn_module, lif_module, num_bits_w=4,num_bits_b=4, num_bits_u=4):
        super(QConvBN2dLIF,self).__init__()

        self.conv_module = conv_module
        self.lif_module = lif_module
        self.bn_module  = bn_module

        self.num_bits_w = num_bits_w
        self.num_bits_b = num_bits_b
        self.num_bits_u = num_bits_u
        
        # Initialize shared scaling factor beta (= alpha in MINT paper, Section IV-C)
        # Formula: 2 * mean(|W|) / sqrt(2^(n-1) - 1):
        # sets an initial quantization range proportional to the weight magnitude.
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()

    def forward(self, x):
        # Quantize weights
        if args.wq:
            if args.share:
                qweight,beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight
            
        # Standard BN, not folded
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
                                        padding=self.conv_module.padding,
                                        dilation=self.conv_module.dilation,
                                        groups=self.conv_module.groups)
        x = self.bn_module(x)
        
        # Pass shared scaling factor beta to LIF neuron
        if args.share:
            s = self.lif_module(x, args.share, beta, bias=0)
        else:
            s = self.lif_module(x, args.share, 0, bias=0)
        #return spikes from LIF neuron
        return s

"""
QConvBN2d — Quantized Conv + BN only (no LIF)
Used for shortcut/residual connections in ResNet-style architectures,
where the branch just needs to match dimensions — no spike generation needed.
"""
class QConvBN2d(nn.Module):
    def __init__(self, conv_module, bn_module, num_bits_w=4,num_bits_u=4,short_cut=False):
        super(QConvBN2d,self).__init__()

        self.conv_module = conv_module
        self.bn_module  = bn_module

        self.num_bits_w = num_bits_w
        self.num_bits_u = num_bits_u
        self.short_cut = short_cut
        
        # Initialize shared scaling factor beta (= alpha in MINT paper, Section IV-C)
        # Formula: 2 * mean(|W|) / sqrt(2^(n-1) - 1):
        # sets an initial quantization range proportional to the weight magnitude.
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()

    def forward(self, x):
        # Quantize weights if enabled
        if args.wq:
            if args.share:
                qweight,beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight
            
        # Conv → BN, output is a feature map (not spikes)
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
                                        padding=self.conv_module.padding,
                                        dilation=self.conv_module.dilation,
                                        groups=self.conv_module.groups)
        # Apply BatchNorm
        x = self.bn_module(x)

        return x