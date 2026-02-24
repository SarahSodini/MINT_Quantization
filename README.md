# MINT_Quantization

## TODO:

I will clean up the codes soon...

## Notice:

I found the code to have some errors when using different PyTorch versions. I will solve the problem later.
For now, please run the code using PyTorch with version 1.13.0. This version is tested to be working. Thanks.

## Citing

If you find MINT is useful for your research, please use the following bibtex to cite us,

```
@inproceedings{yin2024mint,
  title={MINT: Multiplier-less INTeger Quantization for Energy Efficient Spiking Neural Networks},
  author={Yin, Ruokai and Li, Yuhang and Moitra, Abhishek and Panda, Priyadarshini},
  booktitle={2024 29th Asia and South Pacific Design Automation Conference (ASP-DAC)},
  pages={830--835},
  year={2024},
  organization={IEEE}
}
```

## Overview of implementation

The codebase is organized into modules for quantized neural networks, spiking neuron models, quantization utilities, datasets, and training routines. Key files for the MINT method and its formulas are: quant_net.py, quant_resnet.py, network_utils.py, spike_related.py, and training_utils.py.

1. Quantization Formulas (Eq. 3 & 4 in the paper):

- Implemented in training_utils.py: w_q, u_q, and b_q functions.
- Used throughout network_utils.py and in the quantized model layers (e.g., QConv2dLIF, QConvBN2dLIF).

2. LIF Neuron Dynamics (Eq. 1 & 2):

- Implemented in spike_related.py: LIFSpike class, especially in its forward method.
- Handles membrane potential update, spike generation, and reset.

3. Surrogate Gradient for Spiking (Eq. 7):

- Implemented in both training_utils.py (Firing class) and spike_related.py (ZIF class).
- Used for backpropagation through the non-differentiable spike function.

4. Model Composition (Section 4):

- quant_net.py: Defines VGG-like SNNs using quantized convolutional layers and LIF neurons.
- quant_resnet.py: Defines ResNet-like SNNs with quantized blocks and LIF neurons.
- Both use the quantized layers and LIF modules to realize the MINT method.

5. Integration of Quantization and Spiking:

- network_utils.py: Contains quantized convolutional layers that integrate quantization and LIF neuron logic.
- These are used as building blocks in the main models.
  Summary Table:
  Paper Section / Formula Code Location(s) Description
  Eq. 1, 2 (LIF) spike_related.py (LIFSpike) Membrane update, spike, reset
  Eq. 3, 4 (Quantization) training_utils.py (w_q, u_q, b_q), network_utils.py Quantization of weights, potentials
  Eq. 7 (Surrogate Grad) training_utils.py (Firing), spike_related.py (ZIF) Surrogate gradient for spikes
  Model Architecture quant_net.py, quant_resnet.py SNNs with quantized layers and LIF neurons
  If you want a more detailed mapping for a specific formula or section, or want comments in additional files, let me know!
