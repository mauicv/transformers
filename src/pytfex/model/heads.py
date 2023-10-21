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
        b, _, _ = patches.shape
        h, w = self.img_size
        c = self.in_channels
        p_h, p_w = self.patch_size
        patches = patches.view(b, -1, c * p_h * p_w).transpose(1, 2)
        reconstructed = torch.nn.functional.fold(
            patches,
            output_size=(h, w), 
            kernel_size=(p_h, p_w), 
            stride=(p_h, p_w)
        )
        return reconstructed

