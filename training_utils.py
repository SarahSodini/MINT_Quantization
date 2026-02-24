import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


"""
Class for alternative surrogate gradient (not used in main code)
uses a rectangular window centered at 0.5 (the threshold) instead of triangular surrogate gradient.
"""
class Firing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        out = ((inp)>0).float() # fire if input>0
        ctx.save_for_backward(inp)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Rectangular surrogate: gradient = 1 within ±1 of threshold (0.5), 0 elsewhere
        grad = grad_input*torch.where(torch.abs(input-0.5)<1, 1, 0)
        return grad, None, None



"""
integer-only firing condition for MINT inference (threshold comparison in Eq. 9)
Fires a spike if H >= ceil(vth / beta), where ceil(vth/beta) = θ (integer threshold). 
Not used during training (LIFSpike handles that); intended for deployment.
"""
def lif_forward(H, th, beta):
    out = torch.zeros_like(H).cuda()
    out[H >= torch.ceil(th/beta)] = 1
    return out


"""
Quantization functions w_q, u_q, b_q (eq. 10)
All three follow the same pattern:
  1. tanh: smoothly bounds values to (-1, 1) 
  2. clamp(w/alpha): normalize to [-1, 1] using the scaling factor
  3. scale to integer levels: multiply by (2^(b-1) - 1)
  4. STE round: round to nearest integer in forward, straight-through in backward
  5. scale back to original range
"""
def w_q(w, b, alpha):
    #Quantize weights using shared scaling factor alpha
    #Returns quantized weights AND alpha for shared scaling
    w = torch.tanh(w)  # restrict to [-1, 1]
    w = torch.clamp(w/alpha,min=-1,max=1)
    w = w*(2**(b-1)-1)  # scale to quantization levels
    w_hat = (w.round()-w).detach()+w  # STE: straight-through estimator
    return w_hat*alpha/(2**(b-1)-1), alpha

def u_q(u, b, alpha):
    # quantize membrane potential with shared scaling alpha
    #Returns quantized membrane potential AND alpha for shared scaling
    u = torch.tanh(u)
    u = torch.clamp(u/alpha,min=-1,max=1)
    u = u*(2**(b-1)-1)
    u_hat = (u.round()-u).detach()+u
    return u_hat*alpha/(2**(b-1)-1)

def b_q(w, b):
    # Quantize with independent per-tensor scale (not shared).
    # Used when args.share=False => each layer quantizes with its own alpha,
    # derived from the max absolute value of the tensor.
    w = torch.tanh(w)
    alpha = w.data.abs().max()
    w = torch.clamp(w/alpha,min=-1,max=1)
    w = w*(2**(b-1)-1)
    w_hat = (w.round()-w).detach()+w
    return w_hat*alpha/(2**(b-1)-1)


"""Inference-time quantization (no STE)"""
def w_q_inference(w, b, alpha):
    w = torch.tanh(w)
    w = torch.clamp(w/alpha,min=-1,max=1)
    w = w*(2**(b-1)-1)
    w_hat = w.round()
    return w_hat, alpha/(2**(b-1)-1)

def b_q_inference(w, b):
    w = torch.tanh(w)
    alpha = w.data.abs().max()
    w = torch.clamp(w/alpha,min=-1,max=1)
    w = w*(2**(b-1)-1)
    w_hat = w.round()
    return w_hat, alpha/(2**(b-1)-1)

"""
Miscellaneous helpers
"""
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def adjust_learning_rate(optimizer, cur_epoch, max_epoch):
    if (
        cur_epoch == (max_epoch * 0.5)
        or cur_epoch == (max_epoch * 0.7)
        or cur_epoch == (max_epoch * 0.9)
    ):
        for param_group in optimizer.param_groups:
            param_group["lr"] /= 10

"""
Evaluation functions
All test functions sum the per-timestep outputs before computing accuracy:
averages the network's "vote" across all timesteps (rate coding).
"""
def test(model, test_loader, criterion):
    # Standard top-1 accuracy evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            output = sum(model(data))
            # print(type(output))
            _,idx = output.data.max(1, keepdim=True)  # get the index of the max log-probability
            correct += idx.eq(target.data.view_as(idx)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy

def top_k_accuracy(outputs, targets, k=5):
    # Compute top-k accuracy for a single batch
    _, top_pred = outputs.topk(k, 1, True, True)
    top_pred = top_pred.t()
    correct = top_pred.eq(targets.view(1, -1).expand_as(top_pred))
    top_k_acc = correct[:k].view(-1).float().sum(0, keepdim=True) / targets.size(0)
    return top_k_acc.item()

def test_5(model, test_loader, criterion):
    # Top-5 accuracy evaluation — used for TinyImageNet (200 classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    correct = 0
    top5_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            output = sum(model(data))
            test_loss += criterion(output, target).item()  # Accumulate the loss
            _, idx = output.data.topk(5, 1)  # Get the indices of the top-5 predictions
            top5_correct += torch.sum(idx == target.view(-1, 1).expand_as(idx)).item()

    top5_accuracy = 100. * top5_correct / len(test_loader.dataset)

    return top5_accuracy


"""
accumulates spike count and neuron count for spike rate computation.
Divides by 8 to normalize per-timestep (assumes T=8).
"""
def computing_firerate(module, inp, out):
    fired_spikes = torch.count_nonzero(out)
    module.spikerate += fired_spikes/8.0
    module.num_neuron += np.prod(out.shape[1:len(out.shape)])/8.0

"""
Evaluates top-1 accuracy AND average spike rate across all QConv2dLIF layers.
Spike rate measures how sparse the network's activations are: lower is more
energy-efficient on neuromorphic hardware (fewer synaptic operations)
"""
def test_spa(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    overall_nueron = 0
    overall_spike = 0

    # Register hooks on all LIF modules inside QConv2dLIF layers
    neuron_type = 'QConv2dLIF'
    for name, module in model.named_modules():
        if neuron_type in str(type(module)):
            module.lif_module.register_forward_hook(computing_firerate)
            module.lif_module.spikerate = 0
            module.lif_module.num_neuron = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            output = sum(model(data))
            # print(type(output))
            _,idx = output.data.max(1, keepdim=True)  # get the index of the max log-probability
            correct += idx.eq(target.data.view_as(idx)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)

    # Aggregate spike counts across all monitored layers
    for name, module in model.named_modules():
        if neuron_type in str(type(module)):
            overall_nueron += module.lif_module.num_neuron/len(test_loader)
            overall_spike += module.lif_module.spikerate/len(test_loader.dataset)
            # print(overall_nueron)
            # print(module.spikerate/len(test_loader.dataset))
            # print(module.spikerate)
    print("Overall spike rate:", overall_spike/overall_nueron)


    return accuracy,overall_spike/overall_nueron

"""
Plot histogram of membrane potential values for layer l at timestep t.
Used to visualize quantization effects on the potential distribution.
"""
def get_u_distribution(data,l,t,color):
    bins = 128
    i_max = 10
    i_min = -10
    step = (i_max-i_min)/(bins)
    x = np.arange(i_min,i_max,step)
    plt.bar(x, data, align='center', color='#1E97B0')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title('Frequency')
    plt.show()
    plt.savefig(f'./u_fig/hist_u_layer{l}_{t}.pdf', bbox_inches='tight', pad_inches=0.1)


# def get_w_distribution(data,layer_i,e):
    
#     # print("dist tensor", hist)
#     bins = 10
#     hist = torch.histc(data,bins).cpu()
#     i_max = ((torch.max(data).cpu()).item())
#     i_min = ((torch.min(data).cpu()).item())
#     step = (i_max-i_min)/(bins)
#     if step != 0:
#         x = np.arange(i_min,i_max,step)
#         plt.bar(x, hist, align='center', color=['forestgreen'])
#         plt.xlabel('Bins')
#         plt.ylabel('Frequency')
#         plt.title('Frequency')
#         plt.savefig(f'./w_fig/hist_w_{layer_i}_{e}.pdf', bbox_inches='tight', pad_inches=0.1)
#     else:
#         print(f'W are all zeros at layer:{layer_i} at epoch {e}')



"""
Utility class for tracking running averages (e.g., loss, accuracy)
"""
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # Reset all statistics
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # Update statistics with new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

