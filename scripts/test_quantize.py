import argparse

import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from neural_nets import QuantMLP, ConvNet

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for testing post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='convnet_mnist_quant_test.th')
    # parser.add_argument('--filename', help='filename', type=str, default='convnet_mnist.th')
    args = parser.parse_args()

    state_dict = torch.load('../saved_models/' + args.filename)
    # saved_stats = torch.load('../saved_models/' + args.filename)
    # state_dict = saved_stats['state_dict']
    
    quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))
    quant_nn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))
    quant_modules.initialize()


    # model = QuantMLP(in_dim=28 * 28, h_size=h_size, out_dim=10)
    model = ConvNet(out_dim=10).to('cuda')
    model.load_state_dict(state_dict)

    mnist_testset = datasets.MNIST(root='../data', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))

    test_loader = DataLoader(mnist_testset, batch_size=len(mnist_testset.data), num_workers=1, shuffle=False)
    # test_loader = DataLoader(mnist_testset, batch_size=1, num_workers=1, shuffle=False)

            # Disable calibrators
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
    with torch.no_grad():
        acc = 0
        for samples, labels in test_loader:
            logits = model(samples.to('cuda'))
            preds = torch.argmax(logits, dim=1)
            acc += (preds == labels.to('cuda')).sum()
            print(preds)
            print(labels)
            break

    print(f"Accuracy: {(acc / len(mnist_testset.data)) * 100.0:.3f}%")