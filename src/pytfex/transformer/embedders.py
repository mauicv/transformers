import torch
from typing import Tuple, Union
from pytfex.utils import make_tuple


class MultiEmbedder(torch.nn.Module):
    def __init__(self, embedders: Tuple[torch.nn.Module]):
        super(MultiEmbedder, self).__init__()
        self.embedders = torch.nn.ModuleList(embedders)

    def forward(self, x, **kwargs):
        return sum([embedder(x, **kwargs) for embedder in self.embedders])


class TokenEmbedder(torch.nn.Module):
    def __init__(
            self,
            dictionary_size: int=None,
            hidden_dim: int=None,
        ):
        super(TokenEmbedder, self).__init__()

        self.tok_emb = torch.nn.Embedding(
            dictionary_size,
            hidden_dim
        )

    def forward(self, x, **kwargs):
        x = self.tok_emb(x)
        return x


class PositionEmbedder(torch.nn.Module):
    def __init__(
            self, 
            num_positions: int=None,
            hidden_dim: int=None,
        ):
        super(PositionEmbedder, self).__init__()

        self.pos_emb = torch.nn.Embedding(
            num_positions,
            hidden_dim
        )

    def forward(self, x, kv_cache=None, **kwargs):
        if kv_cache is None:
            positions = (torch
                .arange(0, x.shape[1])
                .expand(x.shape[0], -1)
                .to(x.device)
            )
        else:
            leading_position = min(x.shape[1] + kv_cache.size(), self.pos_emb.num_embeddings) - x.shape[1] 
            positions = (torch
                .arange(leading_position, leading_position + x.shape[1])
                .expand(x.shape[0], -1)
                .to(x.device)
            )
        x = self.pos_emb(positions)
        return x


class LinearEmbedder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = torch.nn.Linear(
            self.input_dim,
            self.hidden_dim
        )

    def forward(self, x, **kwargs):
        return self.linear(x)


class PatchEmbedder(torch.nn.Module):
    # Taken from
    # https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
    def __init__(
            self,
            img_size: Union[Tuple[int], int],
            patch_size: Union[Tuple[int], int],
            overlap: Union[Tuple[int], int],
            hidden_dim: int,
            in_channels: int,
        ):
        super().__init__()
        img_size, patch_size, overlap = \
            make_tuple(img_size), make_tuple(patch_size), make_tuple(overlap)
        self.patch_size = patch_size
        self.stride = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])
        self.img_size = img_size
        self.num_patches = (
            ((img_size[0] - patch_size[0]) // self.stride[0] + 1) *
            ((img_size[1] - patch_size[1]) // self.stride[1] + 1)
        )
        
        self.linear = torch.nn.Linear(
            in_channels * patch_size[0] * patch_size[1],
            hidden_dim
        )

    def forward(self, X, **kwargs):
        return self.linear(X)

    def get_patches(self, images):
        b, c, _, _ = images.shape
        p_h, p_w = self.patch_size
        stride_h, stride_w = self.stride
        patches = images.unfold(2, p_h, stride_h).unfold(3, p_w, stride_w)
        patches = patches.contiguous().view(b, c, -1, p_h, p_w)
        patches = patches.transpose(1, 2)
        patches = patches.reshape(b, -1, c * p_h * p_w)
        return patches
