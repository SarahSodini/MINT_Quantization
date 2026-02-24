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
from training_utils import Firing, w_q, b_q
from network_utils import *
from spike_related import LIFSpike

args = args_config.get_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

"""
BasicBlock
The fundamental building block of the ResNet, adapted for SNNs with MINT
quantization. Implements a residual connection: the input x is added back
to the main path output before the final LIF fires.
Structure:
  x ──► ConvBnLif1 ──► ConvBn2 ──► (+) ──► lif2 ──► out
  └────────────── shortcut ────────────┘
"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, n_w, n_u, n_b, stride=1):
        super(BasicBlock, self).__init__()
        # Quantization bitwidths
        self.num_bits_w = n_w
        self.num_bits_b = n_b
        self.num_bits_u = n_u

        # First quantized Conv-BN-LIF block
        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes)
        lif1 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif1 = QConvBN2dLIF(conv1,bn1,lif1,self.num_bits_w,self.num_bits_b,self.num_bits_u)
        
        # Second quantized Conv-BN block
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes)
        self.ConvBn2 = QConvBN2d(conv2,bn2,self.num_bits_w,self.num_bits_u)
        
        # Fires based on the combined (main path + shortcut) membrane potential.
        self.lif2 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        conv_sh = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        bn_sh = nn.BatchNorm2d(self.expansion*planes)
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QConvBN2d(conv_sh,bn_sh, self.num_bits_w, self.num_bits_u, short_cut=True)
            )

    def forward(self, x):
        # Forward through quantized Conv-BN-LIF, then Conv-BN, add shortcut, then LIF
        out = self.ConvBnLif1(x)
        out = self.ConvBn2(out)
        out += self.shortcut(x)
        out = self.lif2(out, args.share, self.ConvBn2.beta[0], bias=0)
        return out

"""
ResNet
Full ResNet SNN built from BasicBlocks. 
Supports both static image datasets and DVS event-based input.
Ends with a two-layer FC classifier (with a LIF between them).
"""
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, total_timestep =4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.total_timestep = total_timestep

        if args.dataset == 'dvs':
            input_dim = 2
        else:
            input_dim = 3

        # ── Layer 1: NOT quantized ─────────────────────
        # conv+BN runs once on the static image,
        # then direct_lif integrates the result across timesteps.
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        #quantization values
        self.num_bits_u = 16
        self.num_bits_w = 16
        self.num_bits_b = 8

        print("ResNet-basic-block weight bits: ", self.num_bits_w)
        print("ResNet-basic-block potential bits: ", self.num_bits_u)

        # ── DVS stem: quantized Conv+BN+LIF (runs inside time loop) ──────────
        conv1dvs = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        bn1dvs = nn.BatchNorm2d(64,affine=True)
        lif1dvs =  LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif1 = QConvBN2dLIF(conv1dvs,bn1dvs,lif1dvs,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.direct_lif = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=False)

        # ── Three residual stages ─────────────────────────────────────────────
        # Stacks `num_blocks[i]` BasicBlocks
        self.layer1 = self._make_layer(block, 128, num_blocks[0], self.num_bits_w, self.num_bits_u, self.num_bits_b, stride=2)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], self.num_bits_w, self.num_bits_u, self.num_bits_b, stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], self.num_bits_w, self.num_bits_u, self.num_bits_b, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pool: collapses any spatial size down to 1x1
        
        # ── Two-layer classifier ──────────────────────────────────────────────
        # ResNet uses FC → LIF → FC.
        # lif_fc is NOT quantized (quant_u=False) — the final classification
        # stage is kept at full precision for accuracy.
        self.fc1 = nn.Linear(512, 256)
        self.lif_fc = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=False)
        self.fc2 = nn.Linear(256, num_classes)

    """
    Stack num_blocks BasicBlocks into sequential layer where first block uses given stride and 
    all subsequent ones use stride = 1
    """
    def _make_layer(self, block, planes, num_blocks, n_w, n_u, n_b, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, n_w, n_u, n_b, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    """
    Reset membrane potentials of all LIF neurons
    """
    def reset_dynamics(self):
        for m in self.modules():
            if isinstance(m,QConv2dLIF):
                m.lif_module.reset_mem()
            elif isinstance(m,QConvBN2dLIF):
                m.lif_module.reset_mem()
            elif isinstance(m,LIFSpike):
                m.reset_mem()
        self.direct_lif.reset_mem()
        self.lif_fc.reset_mem()
        return 0

    """ 
    Initialize weights
    """
    def weight_init(self):
        for m in self.modules():
            if isinstance(m,QConvBN2dLIF):
                nn.init.kaiming_uniform_(m.conv_module.weight)
                nn.init.kaiming_uniform_(m.bn_module.weight)
            elif isinstance(m,QConvBN2d):
                nn.init.kaiming_uniform_(m.conv_module.weight)
                nn.init.kaiming_uniform_(m.bn_module.weight)
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


    def forward(self, x):
        u_out = []
        self.reset_dynamics()
        if args.dataset != 'dvs':
            #static encoding: run conv+BN once outside the time loop
            static_x = self.bn1(self.conv1(x))

        for t in range(self.total_timestep):
            if args.dataset == 'dvs':
                # DVS: extract the t-th event frame and process it fresh each timestep
                out = x[:,t].to(torch.float32).cuda()
                out = self.ConvBnLif1(out)
            else:
                # Static: feed the same feature map to the LIF every timestep;
                # it accumulates and fires more as t increases
                out = self.direct_lif.direct_forward(static_x,False,0)

            #pass through
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)

            # Collapse spatial dims, flatten, then classify through FC → LIF → FC
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.lif_fc(self.fc1(out),False,0,bias=0)
            out = self.fc2(out)

            u_out += [out] # collect per-timestep logits; averaged at loss computation

        return u_out

# Factory function for ResNet18 SNN
def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

# Factory function for custom ResNet19 SNN
def ResNet19(num_classes, total_timestep):
    return ResNet(BasicBlock, [3,3,2], num_classes, total_timestep)

# Factory function for ResNet34 SNN
def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

# Factory function for ResNet50 SNN (uses Bottleneck blocks)
def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

# Factory function for ResNet101 SNN (uses Bottleneck blocks)
def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

# Factory function for ResNet152 SNN (uses Bottleneck blocks)
def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

# Simple test function to check ResNet output shape
def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

