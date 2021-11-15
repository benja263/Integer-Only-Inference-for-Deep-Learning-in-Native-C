import argparse

import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from run_nn import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for testing post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fxp', help='to run with fixed-point instead of float', action='store_true')

    args = parser.parse_args()

    mnist_testset = datasets.MNIST(root='../data', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))

    print(f'Eval Model on Test Samples')
    
    test_loader = DataLoader(mnist_testset, batch_size=len(mnist_testset.data), num_workers=1, shuffle=False)
    # test_loader = DataLoader(mnist_testset, batch_size=1, num_workers=1, shuffle=False)
    
    fxp = False

    c_lib = load_c_lib()
    for samples, labels in test_loader:
        if fxp:
            samples = (samples * (2 ** 16)).round()
            # print(samples)
            preds = run_fxp_mlp(samples, c_lib).astype(int)
        else:
            preds = run_mlp(samples, c_lib).astype(int)
        acc = (torch.from_numpy(preds) == labels).sum()
        break

    print(f"Accuracy: {(acc / len(samples)) * 100.0:.3f}%")