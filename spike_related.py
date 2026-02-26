import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training_utils import *
import tracemalloc
import gc


"""
ZIF — Zero-Is-Fire surrogate gradient (defined but not used)
"""
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




"""
LIF neuron with optional MINT quantization of the membrane potential.
Maintains persistent state across timesteps — call reset_mem() between input samples.

forward()        — standard path, accepts a bias argument for API compatibility
direct_forward() — identical but no bias argument, used for the static encoding first layer
"""
class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, leak=0.5, gamma=1.0, soft_reset=True, quant_u=False, num_bits_u=4):
        super(LIFSpike, self).__init__()
        self.quant_u = quant_u          #membrane potential quantization flag
        self.num_bits_u = num_bits_u    #membrane potential bit width for quantization
        self.thresh = thresh            #firing threshold (θ in the paper)
        self.leak = leak                #membrane potential decay factor (λ in the paper)
        self.gamma = gamma              #surrogate gradient sharpness (γ)
        self.soft_reset = soft_reset    #if True, use soft reset; else, hard reset
        self.membrane_potential = 0     #U: persistent state across timesteps

    """Reset membrane potential to zero between input samples to prevent state bleeding"""
    def reset_mem(self):
        self.membrane_potential = 0

    """
    Implements quantized LIF update and firing
    s: incoming signal from convolution layer (corresponds to W*S in paper eq. 9)
    share: flag to use shared scaling factor beta
    beta: scaling factor
    bias: bias term
    """
    def forward(self, s, share, beta, bias):
        # ── Residual membrane potential (Eq. 6) ─────────────────────────────────────────────────
        # conv output s=W*S is added to the membrane potential from last timestep (τ*U(t-1)).
        # W*S was already computed by the conv layer, membrane_potential=τ*U(t-1) was stored at the end of the last timestep.
        H = s + self.membrane_potential
        
        # ── Fire (Eq.9) ────────────────────────────
        # spike = 1 if H > thresh, else 0
        # grad: triangular surrogate gradient centered at thresh to resolve zero grad problem
        grad = ((1.0 - torch.abs(H-self.thresh)).clamp(min=0)) # triangular surrogare gradient
        ############s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad -- the other line might be a bug!!!##############
        s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad.detach()

        # ── Reset membrane potential ─────────────────────────────────────────────────────
        if self.soft_reset:
            # Soft reset (default):
            #   s=1 -> U = (H - thresh) * leak: keeps within threshold
            #   s=0 -> U = H * leak: decays
            U = (H - s*self.thresh)*self.leak
        else:
            # Hard reset: force U to 0 when spike fires.
            #   s=1 -> U = 0     
            #   s=0 -> U = H * leak: decay
            U = H*self.leak*(1-s)
        
        # ── Quantize membrane potential (Eq. 10) ────────────────────────
        # If quant_u=True, quantize U before storing it as the next timestep's state.
        # u_q: quantization using the shared scaling factor beta 
        # b_q: Quantization without shared scaling factor beta
        if self.quant_u:
            if share:
                self.membrane_potential = u_q(U,self.num_bits_u,beta)
            else:
                self.membrane_potential= b_q(U,self.num_bits_u)
        else:
            self.membrane_potential = U
        
        #return binary spike output
        return s
    
    """
    Same as forward() without the bias argument.
    Used for the first layer where the same static conv feature map is fed every timestep.
    """
    def direct_forward(self, s, share, beta):
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