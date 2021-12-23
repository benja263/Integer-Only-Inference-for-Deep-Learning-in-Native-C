# Uniform Quantization for Integer-Only Inference using Native C
A repository containing Native C-code implementation of a convolutional neural network and multi-layer preceptron (MLP) models.  
The repository contains scripts for training these models with PyTorch, writing the relevant parameters in C, and running interfacing the C code for inference via C-types.  

The ideas presented in this tutorial were used to quantize and write an inference only C code to deploy a deep reinforcement learning algorithm on a network interface card (NIC) in Tessler et al. 2021. 

# Requirements
Quantization is based on Nvidia's pytorch-quantization, which is part of TensorRT.  
https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization
pytorch-quantization allows for more sophisticated quantization methods then what is presented here. For more details see Hao et al. 2020.

**NOTE** pytorch-quantization requires a GPU and will not work without it

## C-code
The c-code is structured to have separate files for the MLP model and the ConvNet model.  
C-code is located within the `src` directory in which:
- nn_math - source and header files contain relevant mathematical functions
- nn - source and header files contain relevant layers to create the neural network models
- mlp - source and header files contains the MLP architecture to run for inference
- convnet - source and header files contains the ConvNet architecture to run for inference
- mlp_params - source and header files are generated via `scripts/create_mlp_c_params.py` and contains network weights, scale factors, and other relevant constants for the MLP model
- convnet_params - source and header files are generated via `scripts/create_convnet_c_params.py` and contains network weights, scale factors, and other relevant constants for the ConvNet model
### Complilation
The repository was tested using gcc
To compile and generate a shared library that can be called from Python using c-types run the following commands:
#### MLP
```
gcc -Wall -fPIC -c mlp_params.c mlp.c nn_math.c nn.c
gcc -shared mlp_params.o mlp.o nn_math.o nn.o -o mlp.so
```
#### ConvNet
```
gcc -Wall -fPIC -c convnet_params.c convnet.c nn_math.c nn.c
gcc -shared convnet_params.o convnet.o nn_math.o nn.o -o convnet.so
```
## Scripts
- `src/train_mlp.py` and `src/train_convnet.py` are used to train an MLP/ConvNet model using PyTorch
- `src/quantize_with_package.py` is used to quantize the models using pytorch-quantization package
- `src/create_mlp_c_params.py` and `src/create_convnet_c_params.py` creates the header and source C files with relevant constants (network parameters, scale factors, and more) required to run the C-code.
- `src/test_mlp_c.py` and `src/test_convnet_c.py` run inference on the models using C-types to interface the C-code files from python


## Results
### MLP
```
Training 
Epoch: 1 - train loss: 0.35650 validation loss: 0.20097
Epoch: 2 - train loss: 0.14854 validation loss: 0.13693
Epoch: 3 - train loss: 0.10302 validation loss: 0.11963
Epoch: 4 - train loss: 0.07892 validation loss: 0.11841
Epoch: 5 - train loss: 0.06072 validation loss: 0.09850
Epoch: 6 - train loss: 0.04874 validation loss: 0.09466
Epoch: 7 - train loss: 0.04126 validation loss: 0.09458
Epoch: 8 - train loss: 0.03457 validation loss: 0.10938
Epoch: 9 - train loss: 0.02713 validation loss: 0.09077
Epoch: 10 - train loss: 0.02135 validation loss: 0.09448
Evaluating model on test data
Accuracy: 97.450%
```
```
Evaluating integer-only C model on test data
Accuracy: 97.27%
```
### ConvNet
```
Training
Epoch: 1 - train loss: 0.37127 validation loss: 0.12948
Epoch: 2 - train loss: 0.09653 validation loss: 0.08608
Epoch: 3 - train loss: 0.07089 validation loss: 0.07480
Epoch: 4 - train loss: 0.05846 validation loss: 0.06347
Epoch: 5 - train loss: 0.05044 validation loss: 0.05909
Epoch: 6 - train loss: 0.04567 validation loss: 0.05466
Epoch: 7 - train loss: 0.04071 validation loss: 0.05099
Epoch: 8 - train loss: 0.03668 validation loss: 0.05336
Epoch: 9 - train loss: 0.03543 validation loss: 0.04965
Epoch: 10 - train loss: 0.03164 validation loss: 0.04883
Evaluate model on test data
Accuracy: 98.620%
```
```
Evaluating integer-only C model on test data
Accuracy: 98.58%
```
# References
Wu, H., Judd, P., Zhang, X., Isaev, M., &#38; Micikevicius, P. (2020). <i>Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation</i>. http://arxiv.org/abs/2004.09602
Tessler, C., Shpigelman, Y., Dalal, G., Mandelbaum, A., Kazakov, D. H., Fuhrer, B., Chechik, G., &#38; Mannor, S. (2021). <i>Reinforcement Learning for Datacenter Congestion Control</i>. http://arxiv.org/abs/2102.09337
