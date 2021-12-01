import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes, activation=nn.ReLU):
        super(MLP, self).__init__()
        assert isinstance(hidden_sizes, list) and len(hidden_sizes) > 0
        layer_list = [nn.Linear(in_dim, hidden_sizes[0], bias=False)]
        for i in range(1, len(hidden_sizes)):
            layer_list.extend([activation(),
                               nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=False)]
                              )
        layer_list.extend([activation(), nn.Linear(hidden_sizes[-1], out_dim, bias=False)])
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))

class ConvNet(nn.Module):
    def __init__(self, out_dim, activation=nn.ReLU):
        super(ConvNet, self).__init__()


        layer_list = [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=False),
         activation(),
         nn.MaxPool2d(kernel_size=(2, 2)),
         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False),
         activation(),
         nn.MaxPool2d(kernel_size=(2, 2)),
         nn.Flatten(),
         nn.Linear(1600, out_dim, bias=False)
        ]

        self.net = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.net(x)


class QuantMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes, activation=nn.ReLU):
        super(QuantMLP, self).__init__()
        assert isinstance(hidden_sizes, list) and len(hidden_sizes) > 0
        layer_list = [nn.Linear(in_dim + 1, hidden_sizes[0], bias=False)]
        for i in range(1, len(hidden_sizes)):
            layer_list.extend([activation(),
                               nn.Linear(hidden_sizes[i-1] + 1, hidden_sizes[i], bias=False)]
                              )
        layer_list.extend([activation(), nn.Linear(hidden_sizes[-1] + 1, out_dim, bias=False)])
        self.net = nn.ModuleList(layer_list)

    def forward(self, x, amax):
        x = x.flatten(start_dim=1)
        for idx, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                x = self.quant(x, amax[f'net.{idx}.input'])
                print(x)
                print(x.shape)
                # add bias dimension
                x = torch.cat((x, torch.ones((len(x), 1), dtype=torch.int32)), dim=1)
                x = x @ self.net[idx].weight.T.int()
                print('before dequant')
                print(x)
                print(x.shape)
                x = self.dequant(x, amax[f'net.{idx}.weight'] * amax[f'net.{idx}.input'])
            else:
                # activation
                x = layer(x)
        return x

    def quant(self, x, amax):
        scale = 127 / amax
        return torch.clamp(torch.round(x * scale), min=-127, max=127).int()

    def dequant(self, x, amax):
        scale = amax / (127**2)
        return x  * scale


