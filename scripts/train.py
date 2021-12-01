"""
Script for training a simple MLP for classification on the MNIST dataset
"""
import argparse
# from typing import Float

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_nets import MLP, ConvNet
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split


def train_epoch(model:nn.Module, data_loader:DataLoader, optimizer:Adam, loss_fn:nn.CrossEntropyLoss):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        optimizer (Adam)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average training loss on epoch
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


def eval_epoch(model: nn.Module, data_loader:DataLoader, loss_fn:nn.CrossEntropyLoss):
    """
    Evaluate epoch on validation data
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average validation loss 
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
    parser.add_argument('--hidden_sizes', help='hidden layer dimensions', nargs='+', type=int, default=[128, 64])
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=10)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.8)
    parser.add_argument('--network_type', help='which network type to use', type=str, choices=['mlp', 'convnet'], default='convnet')

    args = parser.parse_args()

    mnist_trainset = datasets.MNIST(root='../data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))

    # split training data to train/validation
    split_r = args.train_val_split
    mnist_trainset, mnist_valset = random_split(mnist_trainset, [round(len(mnist_trainset)*split_r), round(len(mnist_trainset)*(1 - split_r))])

    mnist_testset = datasets.MNIST(root='../data', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))]))

    model = MLP(in_dim=28*28, out_dim=10, hidden_sizes=args.hidden_sizes) if args.network_type == 'mlp' else ConvNet(out_dim=10)

    optimizer = Adam(model.parameters())

    loss_fnc = nn.CrossEntropyLoss()

    train_loader = DataLoader(mnist_trainset, batch_size=args.batch_size, num_workers=1, shuffle=True)
    val_loader = DataLoader(mnist_valset, batch_size=args.batch_size, num_workers=1, shuffle=True)
    test_loader = DataLoader(mnist_testset, batch_size=args.batch_size, num_workers=1, shuffle=True)

    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fnc)
        val_loss = eval_epoch(model, val_loader, loss_fnc)
        print(f"Epoch: {epoch  + 1} - train loss: {train_loss:.5f} validation loss: {val_loss:.5f}")

    print('Evaluate model on test data')
    model.eval()
    with torch.no_grad():
        acc = 0
        for samples, labels in test_loader:
            logits = model(samples.float())
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc += (preds == labels).sum()

    print(f"Accuracy: {(acc / len(mnist_testset.data))*100.0:.3f}%")
    torch.save({'state_dict': model.state_dict(),
                'hidden_sizes': args.hidden_sizes,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_acc': acc},
               f'../saved_models/{args.network_type}_mnist.th')
