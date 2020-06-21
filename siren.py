import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import List

# based on code from torch.nn.Linear layer:
# https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
class SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, w0 = 1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = self.register_parameter('bias', None)
        self.w0 = torch.tensor(w0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan = nn.init._calculate_correct_fan(self.weights, 'fan_in')
        bound = math.sqrt(6.0 / fan)
        with torch.no_grad():
            self.weights.uniform_(-bound, bound)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform(self.bias, -bound, bound)            

    def forward(self, x) -> torch.Tensor:
        return torch.sin(F.linear(x, self.w0 * self.weights, self.bias))

class SirenModel(nn.Module):
    def __init__(self, layer_dims : List[int]):
        super().__init__()
        first_layer = SirenLayer(layer_dims[0], layer_dims[1], w0 = 30)
        following_layers = []
        for dim0, dim1 in zip(layer_dims[1:-1], layer_dims[2:]):
            following_layers.append(SirenLayer(dim0, dim1))

        self.layers = nn.ModuleList([first_layer, *following_layers])

    def forward(self, x) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y