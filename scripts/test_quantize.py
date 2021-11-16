import argparse

import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from neural_nets import QuantMLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for testing post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_mnist_quant.th')
    args = parser.parse_args()

    saved_stats = torch.load('../saved_models/' + args.filename)
    state_dict = saved_stats['state_dict']
    h_size = saved_stats['h_size']
    amax = saved_stats['amax']

    model = QuantMLP(in_dim=28 * 28, h_size=h_size, out_dim=10)
    model.load_state_dict(state_dict)

    mnist_testset = datasets.MNIST(root='../data', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))

    # test_loader = DataLoader(mnist_testset, batch_size=len(mnist_testset.data), num_workers=1, shuffle=False)
    test_loader = DataLoader(mnist_testset, batch_size=1, num_workers=1, shuffle=False)

    with torch.no_grad():
        acc = 0
        for samples, labels in test_loader:
            logits = model(samples, amax)
            preds = torch.argmax(logits, dim=1)
            acc += (preds == labels).sum()
            break

    print(f"Accuracy: {(acc / len(mnist_testset.data)) * 100.0:.3f}%")