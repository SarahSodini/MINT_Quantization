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
        # ── Step 1: compute residual membrane potential (left hand side Eq. 9) ─────────────────────────────────────────────────
        # conv output s=W*S is added to the membrane potential from last timestep (τ*U(t-1)).
        # W*S was already computed by the conv layer, τ*U(t-1) was stored at the end of the last timestep.
        H = s + self.membrane_potential
        
        # ── Step 2: Fire (right hand side check eq.9) ────────────────────────────
        # spike = 1 if H > thresh, else 0
        # grad: triangular surrogate gradient centered at thresh to resolve zero grad problem
        #
        # grad peaks at 1 when H==thresh, falls to 0 at ±1.
        # This equation solves zero grad problem by making FORWARD pass return the true binary spike while
        # the BACKWARD pass only sees the smooth surrogate gradient
        #
        #   s = (true_spike - H*grad).detach() + H*grad
        #
        # Forward:  .detach() transforms expression into a constant, so H*grad cancels → returns true_spike:
        #             s = (true_spike - H*grad) + H*grad = true_spike 
        #
        # Backward: PyTorch only differentiates through non-detached terms => (true_spike - H*grad) with zero grad falls away =>
        #             ds/dH = d(H*grad)/dH = grad 
        grad = ((1.0 - torch.abs(H-self.thresh)).clamp(min=0))  
        s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad.detach()


        # Quantized membrane potential update (Eq. 6)
        # ── Step 3: Reset  ─────────────────────────────────────────────────────
        # After firing, reduce the membrane potential to prevent immediate re-firing.
        #
        # Soft reset (default): subtract threshold from H when spike fires.
        #   U = (H - s*thresh) * leak
        #   If s=1: U = (H - thresh) * leak  => keeps the "overflow" above threshold
        #   If s=0: U = H * leak             => just decays
        # Gentler and retains information above the threshold.
        #
        # Hard reset: force U to 0 when spike fires.
        #   U = H * leak * (1 - s)
        #   If s=1: U = 0                    => full reset regardless of H
        #   If s=0: U = H * leak             => just decays
        if self.soft_reset:
            U = (H - s*self.thresh)*self.leak
        else:
            U = H*self.leak*(1-s)
        
        # ── Step 4: Quantize membrane potential (Eq. 10) ────────────────────────
        # If quant_u=True, quantize U before storing it as the next timestep's state.
        # u_q: tanh-bounded quantization using the shared scaling factor beta 
        # b_q: Quantization without shared scaling factor beta
        # 
        # OBS: tanh implemented instead of clamp: smoother bounded variant
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

        # Quantized membrane potential update (Eq. 6)
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