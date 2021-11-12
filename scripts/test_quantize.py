import argparse

import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms
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

    print('Eval Model on 5500 Test Samples')
    rand_numbers = np.random.randint(0, mnist_testset.data.shape[0], 5500)
    samples, labels = mnist_testset.data[rand_numbers], mnist_testset.targets[rand_numbers]

    with torch.no_grad():
        logits = model(samples.float(), amax)
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == labels).sum()

    print(f"Accuracy: {(acc / 5500.0) * 100.0:.3f}%")