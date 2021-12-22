"""
Script for running inference of model in C using ctypes
"""
import argparse

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from src.run_nn import load_c_lib, run_convnet
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for testing post-training quantization of a pre-trained model in C",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)

    args = parser.parse_args()

    mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))

    print(f'Evaluate model on test data')
    
    test_loader = DataLoader(mnist_testset, batch_size=args.batch_size, num_workers=1, shuffle=False)

    # load c library
    c_lib = load_c_lib(library='convnet.so')

    acc = 0
    for samples, labels in test_loader:
        samples = (samples * (2 ** 16)).round() # convert to fixed-point 16
        preds = run_convnet(samples, c_lib).astype(int)
        acc += (torch.from_numpy(preds) == labels).sum()

    print(f"Accuracy: {(acc / len(mnist_testset.data)) * 100.0:.2f}%")
