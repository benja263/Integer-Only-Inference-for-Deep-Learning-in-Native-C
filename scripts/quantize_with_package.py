import sys

sys.path.append('/swgwork/bfuhrer/projects/rl_packages/TensorRT/tools/pytorch-quantization/pytorch_quantization')

import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_nets import MLP, ConvNet
from torch.utils.data import DataLoader
import numpy as np

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules


def collect_stats(model, data_loader, num_bins):
    """Feed data to the network and collect statistic"""
    model.eval()
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module._calibrator._num_bins = num_bins
            else:
                module.disable()

    for batch, _ in data_loader:
        x = batch.float()
        model(x)

        # Disable calibrators
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


def quantize_model_params(model):
    """Quantize layer weights using calculated amax
       and process scale constant for C-code

    Args:
        state_dict (Dict): pytorch model state_dict
        amax (Dict): dictionary containing amax values
    """

    is_mlp = isinstance(model, MLP)

    indices = [0, 2, 4] if is_mlp else [0, 3, 7]
    scale_factor = 127 # 127 for 8 bits

    
    state_dict = dict()

    for layer_idx, idx in enumerate(indices, start=1):
        # quantize all parameters
        weight = model.state_dict()[f'net.{idx}.weight']
        s_w = model.state_dict()[f'net.{idx}._weight_quantizer._amax'].numpy()
        s_x = model.state_dict()[f'net.{idx}._input_quantizer._amax'].numpy()

        scale = weight * (scale_factor / s_w)
        state_dict[f'layer_{layer_idx}_weight'] = torch.clamp(scale.round(), min=-127, max=127).to(int)
        if is_mlp or layer_idx == 3:
            state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].T
        state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].numpy()

        state_dict[f'layer_{layer_idx}_s_x'] = 1.0 / s_x
        state_dict[f'layer_{layer_idx}_s_x_inv'] = s_x / scale_factor
        state_dict[f'layer_{layer_idx}_s_w_inv'] = (s_w / scale_factor).squeeze()
        # state_dict[f'layer_{layer_idx}_s_w_inv'] = (s_w).squeeze()
        

    return state_dict
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='convnet_mnist.th')
    parser.add_argument('--num_bins', help='number of bins', type=int, default=128)

    args = parser.parse_args()
    # load model
    saved_stats = torch.load('../saved_models/' + args.filename)


    
    state_dict = saved_stats['state_dict']
    hidden_sizes = saved_stats['hidden_sizes']

    quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))
    quant_nn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))
    quant_modules.initialize()


    model = MLP(in_dim=28*28, hidden_sizes=hidden_sizes, out_dim=10) if 'mlp' in args.filename else ConvNet(out_dim=10)
    model.load_state_dict(state_dict)
    
    mnist_trainset = datasets.MNIST(root='../data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))

    train_loader = DataLoader(mnist_trainset, batch_size=len(mnist_trainset.data), num_workers=1, shuffle=False)

    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, train_loader, args.num_bins)
        compute_amax(model, method="entropy")

    state_dict = quantize_model_params(model)
    saved_stats['state_dict'] = state_dict

    name = args.filename.replace('.th', '_quant.th')

    torch.save(saved_stats,
            f'../saved_models/{name}')
    torch.save(model.state_dict(),
            f"../saved_models/{name.replace('_quant', '_quant_test')}")