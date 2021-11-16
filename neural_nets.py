import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_size, activation=nn.ReLU):
        super(MLP, self).__init__()
        assert isinstance(h_size, list) and len(h_size) > 0
        layer_list = [nn.Linear(in_dim + 1, h_size[0], bias=False)]
        for i in range(1, len(h_size)):
            layer_list.extend([activation(),
                               nn.Linear(h_size[i-1] +1, h_size[i], bias=False)]
                              )
        layer_list.extend([activation(), nn.Linear(h_size[-1] + 1, out_dim, bias=False)])
        self.net = nn.ModuleList(layer_list)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for i in range(len(self.net)):
            if isinstance(self.net[i], nn.Linear):
                # add bias column in input
                x = torch.cat((x, torch.ones((len(x), 1))), dim=1)
            x = self.net[i](x)
        return x


class QuantMLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_size, activation=nn.ReLU):
        super(QuantMLP, self).__init__()
        assert isinstance(h_size, list) and len(h_size) > 0
        layer_list = [nn.Linear(in_dim + 1, h_size[0], bias=False)]
        for i in range(1, len(h_size)):
            layer_list.extend([activation(),
                               nn.Linear(h_size[i-1] + 1, h_size[i], bias=False)]
                              )
        layer_list.extend([activation(), nn.Linear(h_size[-1] + 1, out_dim, bias=False)])
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


