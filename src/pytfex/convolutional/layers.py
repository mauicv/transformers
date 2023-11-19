import torch.nn as nn
from pytfex.convolutional.torch_modules import get_conv, get_norm
from pytfex.convolutional.torch_modules import get_nonlinearity, ActType


class ConvolutionalLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: ActType='identity'
        ):
        super().__init__()
        self.conv = get_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.norm = get_norm()(out_channels)
        self.activation = get_nonlinearity(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x