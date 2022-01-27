from typing import List

import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_layers: List[int] = (1000, 1000),
        activation=nn.ReLU(),
        input_features: int = 1000,
        normalize: bool = False,
        bias_in_last_layer: bool = True,
    ):
        super(FullyConnected, self).__init__()
        self.normalize = normalize
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.hidden_layers.insert(0, input_features)

        modules = []
        for i in range(len(self.hidden_layers) - 1):
            modules.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            modules.append(nn.BatchNorm1d(self.hidden_layers[i + 1]))
            modules.append(activation)

        modules.append(nn.Linear(self.hidden_layers[-1], num_classes, bias=bias_in_last_layer))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        if self.normalize:
            x = self.model(x)
            return F.normalize(x, dim=1)
        else:
            return self.model(x)


class CustomResnet(nn.Module):
    def __init__(self, model, kernel_size=7, stride=2, channels=3, maxpool1=True):
        super(CustomResnet, self).__init__()
        self.model = model

        if not maxpool1:
            self.model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        conv1_out_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(
            channels, conv1_out_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False
        )

    def forward(self, x):
        return self.model(x)


def load_state_dict(model, checkpoint):
    return model.load_state_dict(checkpoint, strict=True)
