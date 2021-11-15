import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_size, activation=nn.ReLU):
        super(MLP, self).__init__()
        assert isinstance(h_size, list) and len(h_size) > 0
        layer_list = [nn.Linear(in_dim, h_size[0])]
        for i in range(1, len(h_size)):
            layer_list.extend([activation(),
                               nn.Linear(h_size[i-1], h_size[i])]
                              )
        layer_list.extend([activation(), nn.Linear(h_size[-1], out_dim)])
        self.net = nn.ModuleList(layer_list)

    def forward(self, x):
        x = self.net[0](x.flatten(start_dim=1))
        for i in range(1, len(self.net)):
            x = self.net[i](x)
        return x


class QuantMLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_size, activation=nn.ReLU):
        super(QuantMLP, self).__init__()
        assert isinstance(h_size, list) and len(h_size) > 0
        layer_list = [nn.Linear(in_dim, h_size[0])]
        for i in range(1, len(h_size)):
            layer_list.extend([activation(),
                               nn.Linear(h_size[i-1], h_size[i])]
                              )
        layer_list.extend([activation(), nn.Linear(h_size[-1], out_dim)])
        self.net = nn.ModuleList(layer_list)
                # self.net[net_idx].bias.copy_(127 * self.net[net_idx].bias / amax[f'net.{net_idx}.bias'])

    def layer_forward(self, x, idx, amax):
        x = x @ self.net[idx].weight.T.int()
        # print(x)
        x = self.dequant(x, amax[f'net.{idx}.weight'] * amax[f'net.{idx}.input'], double=True)
        # print(x)
        return x + self.dequant(self.net[idx].bias, amax[f'net.{idx}.bias'], double=False)

    def forward(self, x, amax):
        x = x.flatten(start_dim=1)
        for idx in range(0, len(self.net), 2):
            x = self.quant(x, amax[f'net.{idx}.input'])
            print(x)
            x = self.layer_forward(x, idx, amax)
            # print(x)
            if idx + 1 < len(self.net):
                x = self.net[idx + 1](x)
        return x

    def quant(self, x, amax):
        scale = 127 / amax
        return torch.clamp(torch.round(x * scale), min=-127, max=127).int()

    def dequant(self, x, amax, double=False):
        scale = amax / (127**2) if double else amax / 127
        return x  * scale


