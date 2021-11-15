import argparse

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import quantization.calibration as calib
from neural_nets import MLP


def quantize_layer_params(state_dict, percentile=99):
    amax = dict()
    for name, params in state_dict.items():
        print(f"Quantizing: {name}")
        if 'weight' in name:
            # if percentile is not None:
            #     hists = [calib.Histogram() for _ in range(0, len(model.net), 2)]
            # else:
            params_amax, _ = params.max(dim=1)
        else:
            params_amax = params.max()
        amax[name] = params_amax
    return amax

def quant_model_params(state_dict, amax):

    # quantize all parameters
    for name, param in state_dict.items():
        param_amax = amax[name]
        # quantize 
        if 'weight' in name:
            param = param.T
        param = param * (127 / param_amax)
        if 'weight' in name:
            param = param.T
        state_dict[name] = torch.clamp(param.round(), min=-127, max=127).to(int)
    # w = self.quant(self.net[idx].weight.clone().T, amax[f'net.{idx}.weight'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_mnist.th')
    parser.add_argument('--num_bins', help='number of bins', type=int, default=128)

    args = parser.parse_args()
    
    saved_stats = torch.load('../saved_models/' + args.filename)
    state_dict = saved_stats['state_dict']
    h_size = saved_stats['h_size']

    model = MLP(in_dim=28*28, h_size=h_size, out_dim=10)
    model.load_state_dict(state_dict)
    
    amax = quantize_layer_params(state_dict)
    quant_model_params(state_dict, amax)
    
    print('finished parameter quantization')

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
                    hists[idx].fill_hist(x.cpu().numpy())
                    amax[f'net.{idx*2}.input'] = calib.compute_amax_entropy(hists[idx].hist, hists[idx].bin_edges, num_bits=8)
                else:
                    idx += 1
                x = layer(x)

    saved_stats['amax'] = amax
    saved_stats['state_dict'] = state_dict
    # saved_stats['state_dict'] = model.state_dict()
    torch.save(saved_stats,
               f'../saved_models/mlp_mnist_quant.th')



