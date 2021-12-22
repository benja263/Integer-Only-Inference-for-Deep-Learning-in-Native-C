"""
Module containing neural network architectures (MLP and ConvNet)
"""
import torch.nn as nn

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
    def __init__(self, out_dim, channel_sizes, activation=nn.ReLU):
        super(ConvNet, self).__init__()

        def get_output_dim(input_dim, kernel_size, stride):
            output_dim = (input_dim -(kernel_size-1) - 1) / stride
            return int(output_dim + 1)

        output_dim = get_output_dim(get_output_dim(28, kernel_size=3, stride=1), kernel_size=2, stride=2)
        output_dim = get_output_dim(get_output_dim(output_dim, kernel_size=3, stride=1), kernel_size=2, stride=2)
        
        layer_list = [nn.Conv2d(in_channels=1, out_channels=channel_sizes[0], kernel_size=(3, 3), bias=False),
         nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),
         activation(),
         nn.Conv2d(in_channels=channel_sizes[0], out_channels=channel_sizes[1], kernel_size=(3, 3), bias=False),
         nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)),
         activation(),
         nn.Flatten(),
         nn.Linear(output_dim*output_dim*channel_sizes[-1], out_dim, bias=False)
        ]

        self.net = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.net(x)