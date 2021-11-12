import argparse

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from neural_nets import MLP


def train_epoch(model, data_loader, optimizer, loss_fn):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    :param nn.Module model:
    :param DataLoader data_loader:
    :param optimizer:
    :param loss_fn: loss function
    :return:
    """
    model.train(mode=True)
    num_batches = len(data_loader)

    loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()
        logits = model(x)

        batch_loss = loss_fn(logits, y)

        batch_loss.backward()

        optimizer.step()
        loss += batch_loss.item()
    return loss / num_batches


def eval_epoch(model, data_loader, loss_fn):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    :param nn.Module model:
    :param DataLoader data_loader:
    :param loss_fn: loss function
    :return:
    """
    model.eval()
    num_batches = len(data_loader)

    loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred_y = model(x)
            batch_loss = loss_fn(pred_y, y)
            loss += batch_loss.item()
    return loss / num_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--h_size', help='hidden layer dimensions', nargs='+', type=int, default=[1024, 768, 512, 128])
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=10)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)

    args = parser.parse_args()

    mnist_trainset = datasets.MNIST(root='../data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))
    mnist_testset = datasets.MNIST(root='../data', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))

    model = MLP(in_dim=28*28, out_dim=10, h_size=args.h_size)

    optimizer = Adam(model.parameters())

    loss_fnc = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(mnist_trainset, batch_size=args.batch_size, num_workers=1, shuffle=True)
    test_loader = DataLoader(mnist_testset, batch_size=args.batch_size, num_workers=1, shuffle=True)

    for epoch in range(args.num_epochs):
        print(f"Epoch: {epoch}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fnc)
        test_loss = eval_epoch(model, test_loader, loss_fnc)
        print(f"train loss: {train_loss:.5f} test loss: {test_loss:.5f}")

    print('Eval Model on 50 Test Samples')
    rand_numbers = np.random.randint(0, mnist_testset.data.shape[0], 50)
    samples, labels = mnist_testset.data[rand_numbers], mnist_testset.targets[rand_numbers]
    with torch.no_grad():
        logits = model(samples.float())
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == labels).sum()

    print(f"Accuracy: {(acc / 50.0)*100.0:.3f}%")
    torch.save({'state_dict': model.state_dict(),
                'h_size': args.h_size},
               f'../saved_models/mlp_mnist.th')
