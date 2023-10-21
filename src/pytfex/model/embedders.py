import torch
from typing import Tuple, Union
from pytfex.utils import make_tuple


class TokenPositionEmbedder(torch.nn.Module):
    def __init__(
            self, 
            max_sequence_length: int=None,
            dictionary_size: int=None,
            hidden_dim: int=None,
        ):
        super(TokenPositionEmbedder, self).__init__()

        self.pos_emb = torch.nn.Embedding(
            max_sequence_length,
            hidden_dim
        )

        self.tok_emb = torch.nn.Embedding(
            dictionary_size,
            hidden_dim
        )

    def forward(self, x):
        positions = (torch
            .arange(0, x.shape[1])
            .expand(x.shape[0], -1)
            .to(x.device)
        )
        x = self.tok_emb(x) + self.pos_emb(positions)
        return x


class PatchEmbedder(torch.nn.Module):
    # Taken from
    # https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
    def __init__(
            self,
            img_size: Union[Tuple[int], int],
            patch_size: Union[Tuple[int], int],
            hidden_dim: int,
            in_channels: int
        ):
        super().__init__()
        img_size, patch_size = make_tuple(img_size), make_tuple(patch_size)
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size[0] // patch_size[0]) \
            * (img_size[1] // patch_size[1])

        self.pos_emb = torch.nn.Embedding(
            self.num_patches,
            hidden_dim
        )
        
        self.linear = torch.nn.Linear(
            in_channels * patch_size[0] * patch_size[1],
            hidden_dim
        )

    def forward(self, X):
        P = self.linear(X)
        _, l, _ = P.shape
        positions = (torch
            .arange(0, l)
            .expand(P.shape[0], -1)
            .to(P.device)
        )
        return P + self.pos_emb(positions)

    def get_patches(self, images):
        b, c, _, _ = images.shape
        p_h, p_w = self.patch_size
        patches = images.unfold(2, p_h, p_h).unfold(3, p_w, p_w)
        patches = patches.contiguous().view(b, c, -1, p_h, p_w)
        patches = patches.view(b, -1, c * p_h * p_w)
        return patches


class PositionEmbedder(torch.nn.Module):
    def __init__(
        self, 
        number_positions: int,
        hidden_dim: int,
    ):
        super(PositionEmbedder, self).__init__()
        self.pos_emb = torch.nn.Embedding(
            number_positions,
            hidden_dim
        )

    def forward(self, x):
        b, l, _ = x.shape
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        positions = torch.arange(0, l).expand(b, -1).to(device)
        return x + self.pos_emb(positions)
