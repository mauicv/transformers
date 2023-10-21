import torch
from pytfex.utils import make_tuple


class ClassificationHead(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            vocab_size: int,
        ):
        super(ClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.ln_f = torch.nn.LayerNorm(
            self.hidden_dim
        )
        self.linear = torch.nn.Linear(
            self.hidden_dim,
            self.vocab_size,
            bias=False
        )

    def forward(self, x):
        return self.linear(self.ln_f(x))


class InversePatch(torch.nn.Module):
    def __init__(
            self,
            img_size: tuple,
            patch_size: tuple,
            hidden_dim: int,
            in_channels: int,
        ):
        super(InversePatch, self).__init__()
        self.img_size = make_tuple(img_size)
        self.patch_size = make_tuple(patch_size)
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.linear = torch.nn.Linear(
            self.hidden_dim,
            self.in_channels * self.patch_size[0] * self.patch_size[1],
            bias=False
        )

    def forward(self, x):
        x = self.linear(x)
        return x

    def get_images(self, patches):
        return patches.reshape(
            patches.shape[0],
            self.in_channels,
            self.img_size[0],
            self.img_size[1]
        )
