"""
Module for layer quantization
"""
import argparse
from typing import Dict, Float

import quantization.calibration as calib
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_nets import MLP
from torch.utils.data import DataLoader


def get_layer_amax(state_dict:Dict):
    """Get per layer weight amax values via max quantization of layer weights with per-row granulity 

    Args:
        state_dict (Dict): pytorch model state_dict

    Returns:
        [Dict]: dictionary containing amax values
    """
    amax = dict()
    for name, params in state_dict.items():
        print(f"Quantizing: {name}")
        params_amax, _ = params.max(dim=1)
        amax[name.replace('weight', 's_w')] = params_amax
    return amax

def quantize_model_params(state_dict:Dict, amax:Dict):
    """Quantize layer weights using calculated amax

    Args:
        state_dict (Dict): pytorch model state_dict
        amax (Dict): dictionary containing amax values
    """

    # quantize all parameters
    for name, param in state_dict.items():
        # follow paper's naming
        param_amax = amax[name.replace('weight', 's_w')]
        # quantize  and re-transpose such that weights are compatible with pytorch's weight shape
        param = (param.T * (127 / param_amax).T
        state_dict[name] = torch.clamp(param.round(), min=-127, max=127).to(int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_mnist.th')
    parser.add_argument('--num_bins', help='number of bins', type=int, default=128)

    args = parser.parse_args()
    # load model
    saved_stats = torch.load('../saved_models/' + args.filename)
    
    state_dict = saved_stats['state_dict']
    hidden_sizes = saved_stats['hidden_sizes']

    model = MLP(in_dim=28*28, hidden_sizes=hidden_sizes, out_dim=10)
    model.load_state_dict(state_dict)

    amax = get_layer_amax(state_dict)
    quantize_model_params(state_dict, amax)
    
    print('finished model weight quantization\n starting input/activation quantization')
    
    mnist_trainset = datasets.MNIST(root='../data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))

    train_loader = DataLoader(mnist_trainset, batch_size=len(mnist_trainset.data), num_workers=1, shuffle=False)

    hists = [calib.Histogram(num_bins=args.num_bins,) for _ in range(0, len(model.net), 2)]
    with torch.no_grad():
        idx = 0
        for batch, y in train_loader:
            x = batch.float().flatten(start_dim=1)
            for layer in model.net:
                print(layer)
                if isinstance(layer, nn.Linear):
                    # fill histogram object with input data
                    hists[idx].fill_hist(x.cpu().numpy())
                    # calculate amax using entropy
                    amax[f'net.{idx*2}.s_x'] = calib.compute_amax_entropy(hists[idx].hist, hists[idx].bin_edges, num_bits=8)
                    # add bias
                    x = torch.cat((x, torch.ones((len(x), 1))), dim=1)
                else:
                    idx += 1       
                x = layer(x)
                
    # save quantized model weights and amax values
    saved_stats['amax'] = amax
    saved_stats['state_dict'] = state_dict
    torch.save(saved_stats,
               f'../saved_models/mlp_mnist_quant.th')



