import torch.nn as nn
from pytfex.convolutional.resnet import ResnetBlock
from pytfex.convolutional.torch_modules import get_conv, get_norm
from pytfex.convolutional.torch_modules import get_nonlinearity, ActType
from typing import Optional, Literal


class DownSampleBlock(nn.Module):
    def __init__(
            self, 
            in_filters, 
            out_filters, 
            kernel=24, 
            stride=4, 
            padding=12,
            dropout=0.1,
            activation='ELU'):
        super().__init__()
        self.conv = get_conv(
            in_channels=in_filters,
            out_channels=out_filters,
            kernel_size=kernel,
            stride=stride,
            padding=padding
        )
        self.norm = get_norm()(out_filters)
        self.nonlinearity = get_nonlinearity(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return x

    @classmethod
    def default(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, kernel=4, stride=2, padding=1)


class EncoderLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_residual: int
        ) -> None:
        super().__init__()
        self.components = nn.ModuleList()
        for _ in range(num_residual):
            self.components.append(ResnetBlock(
                in_channels=in_channels, 
                out_channels=out_channels,
            ))
            in_channels = out_channels
        self.components.append(DownSampleBlock.default(
            in_channels,
            out_channels
        ))

    def forward(self, x):
        for component in self.components:
            x = component(x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            nc: int, 
            ndf: int,
            layers: list[nn.Module],
            output_layer: Optional[nn.Module]=None
        ):
        super(Encoder, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.input_conv = get_conv(
            in_channels=nc,
            out_channels=ndf,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.input_norm = get_norm()(ndf)
        self.input_activation = get_nonlinearity('ELU')
        self.input_dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList(layers)
        self.output_layer = output_layer

    def forward(self, x):
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)
        for layer in self.layers:
            x = layer(x)
        if self.output_layer:
            x = self.output_layer(x)
        return x

    def loss(self, x, y, layer_inds=None):
        if not layer_inds:
            layer_inds = [i for i in range(self.depth)]
        layer_inds = set(layer_inds)

        batch_size = x.shape[0]
        x = self.input_conv(x)
        y = self.input_conv(y)
        sum = 0

        for ind, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            if ind in layer_inds:
                rx = x.reshape(batch_size, -1)
                ry = y.reshape(batch_size, -1)
                sum = sum + ((rx - ry)**2).mean(-1)
        return sum


# class OutputLayer(nn.Module):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             activation: ActType='identity'
#         ):
#         super().__init__()
#         self.conv = get_conv(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
#         self.norm = get_norm()(out_channels)
#         self.activation = get_nonlinearity(activation)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.activation(x)
#         return x