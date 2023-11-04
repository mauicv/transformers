import torch.nn as nn
from pytfex.convolutional.resnet import ResnetBlock
from pytfex.convolutional.torch_modules import get_conv, get_norm, get_rep_pad, get_upsample
from pytfex.convolutional.torch_modules import get_nonlinearity


class UpSampleBlock(nn.Module):
    def __init__(
            self, 
            in_filters, 
            out_filters, 
            scale_factor, 
            kernel, 
            padding,
            dropout=0.1,
            activation='ELU'):
        super().__init__()
        self.conv = get_conv(
            in_channels=in_filters, 
            out_channels=out_filters,
            kernel_size=kernel,
            padding=0
        )
        self.up = get_upsample()(scale_factor=scale_factor)
        self.rp2d = get_rep_pad()(padding)
        self.norm = get_norm()(out_filters, 1.e-3)
        self.nonlinearity = get_nonlinearity(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up(x)
        x = self.rp2d(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return x

    @classmethod
    def default(cls, in_filters, out_filters):
        return cls(in_filters, out_filters, scale_factor=2,
                   kernel=3, padding=1, dropout=0.1)


class DecoderLayer(nn.Module):
    def __init__(self, in_filters, out_filters, num_residual) -> None:
        super().__init__()
        self.components = nn.ModuleList()
        for _ in range(num_residual):
            self.components.append(ResnetBlock(
                in_channels=in_filters, 
                out_channels=out_filters,
            ))
            in_filters = out_filters
        self.components.append(
            UpSampleBlock.default(in_filters, out_filters)
        )

    def forward(self, x):
        for component in self.components:
            x = component(x)
        return x

class Decoder(nn.Module):
    def __init__(
            self,
            nc, 
            ndf,
            layers,
        ):
        super(Decoder, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.output_conv = get_conv(
            in_channels=ndf,
            out_channels=nc,
            kernel_size=1
        )

        self.layers = layers

    def forward(self, z):
        print(z.shape)
        for layer in self.layers:
            z = layer(z)
        x = self.output_conv(z)
        return x