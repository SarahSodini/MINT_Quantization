import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training_utils import *
import tracemalloc
import gc


# Surrogate gradient function for spiking neurons (see Eq. 7 in the MINT paper)
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward: binary spike if input > 0
        out = (input > 0).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: surrogate gradient for non-differentiable spike
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input * ((1.0 - torch.abs(input)).clamp(min=0))
        return grad_input




    # Leaky Integrate-and-Fire (LIF) neuron module
    # Implements the LIF neuron dynamics (see Eq. 1 and 2 in the MINT paper)
class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, leak=0.5, gamma=1.0, soft_reset=True, quant_u=False, num_bits_u=4):
        """
        Implements LIF neuron with optional quantization of membrane potential.
        Args:
            thresh: firing threshold (θ in the paper)
            leak: membrane potential decay factor (λ in the paper)
            gamma: surrogate gradient sharpness (γ)
            soft_reset: if True, use soft reset; else, hard reset
            quant_u: if True, quantize membrane potential
            num_bits_u: bitwidth for quantization
        """
        super(LIFSpike, self).__init__()
        self.quant_u = quant_u
        self.num_bits_u = num_bits_u
        self.thresh = thresh
        self.leak = leak
        self.gamma = gamma
        self.soft_reset = soft_reset
        self.membrane_potential = 0

    def reset_mem(self):
        # Reset membrane potential to zero
        self.membrane_potential = 0

    def forward(self, s, share, beta, bias):
        # Implements LIF update and spike generation (see Eq. 1 and 2)
        # s: input current, share: use shared quantization, beta: scaling, bias: bias term
        # Compute new membrane potential
        H = s + self.membrane_potential
        # Surrogate gradient for spike generation (Eq. 7)
        grad = ((1.0 - torch.abs(H-self.thresh)).clamp(min=0))
        s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad.detach()
        # Update membrane potential with soft or hard reset (Eq. 2)
        if self.soft_reset:
            U = (H - s*self.thresh)*self.leak
        else:
            U = H*self.leak*(1-s)
        # Optional quantization of membrane potential (Eq. 4)
        if self.quant_u:
            if share:
                self.membrane_potential = u_q(U,self.num_bits_u,beta)
            else:
                self.membrane_potential= b_q(U,self.num_bits_u)
        else:
            self.membrane_potential = U
        return s
    
    def direct_forward(self, s, share, beta):
        # Direct forward for static input (no bias)
        H = s + self.membrane_potential
        grad = ((1.0 - torch.abs(H-self.thresh)).clamp(min=0))
        s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad.detach()
        if self.soft_reset:
            U = (H - s*self.thresh)*self.leak
        else:
            U = H*self.leak*(1-s)
        if self.quant_u:
            if share:
                self.membrane_potential = u_q(U,self.num_bits_u,beta)
            else:
                self.membrane_potential= b_q(U,self.num_bits_u)
        else:
            self.membrane_potential = U
        return s