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
## Files overview
### args_config
Argument configuration file for models and training
sets default:
- batch size = 256
- learning rate=1e-3
- epoch = 200
- 4 dataloader workers
- res19 default model architecture out of [vgg9,vgg16,res19]
- dvs default dataset out of [cifar10,svhn,tiny,dvs]
- adam default optimizer out of [adam, sgd]
- leak_mem: (τ) default 0.5
- th: firing threshold default 0.5
- rst: reset type, default hard out of [hard,soft]
- T: nr of timesteps per sample, default 10

quantization arguments: 
- uq: flag for enabling uniform quantization
- bq: flag bias quantization
- wq: flag weight quantization
- share: flag shared scaling factor
- sft_rst: flag soft reset
- conv_b: flag bias in conv block
- bn_a: flag for affine parameters in BatchNorm2d

### cifar_dvs_dataset
Prepares dvs data for training

### network utils
Defines the network building blocks architectures and initializes them according to args flags. Implements different combinations of quantized convolution, batch normalization (folded and not folded) and LIF across 4 classes: 
- QConv2dLIF — Conv + LIF only(simplest)
- QConvBN2dLIF — Conv + BN (separate, not folded) + LIF
- QFConvBN2dLIF — Conv + BN (folded into conv) + LIF (most optimized)
- QConvBN2d — Conv + BN, no LIF (for residual shortcuts)

### quant_net
Defines 3 VGG-style SNN model classes (Q_ShareScale_VGG9, Q_ShareScale_VGG16, Q_ShareScale_Fold_VGG16) that stack MINT-quantized Conv-BN-LIF blocks from network utils into full network architectures. Handles both static image datasets and DVS event-based input

### quant_resnet
Defines the MINT-quantized ResNet architecture (BasicBlock and ResNet) with residual connections, where each block applies a quantized Conv-BN-LIF from network utils into full network architectures. Handles both static image datasets and DVS event-based input

### spike_related
implements core LIF spiking neuron with an inline surrogate gradient for backpropagation, soft/hard reset modes and optional MINT quantization of the membrane potential. The neuron maintains persistent state across timesteps and must be reset between input samples

**eq. 6, 9, 10**

### train_snn
Main training script: builds the dataset, model, and optimizer from command-line arguments, then runs the train/test loop, saving the best checkpoint.

### training_utils
Utility functions for training and evaluation: quantization functions (w_q, u_q, b_q), test routines (top-1, top-5, spike rate), the surrogate gradient Firing class, and miscellaneous helpers for learning rate scheduling and checkpoint management.

**eq. 9, 10**