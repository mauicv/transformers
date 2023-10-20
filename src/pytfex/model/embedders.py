import torch
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
            img_size: int,
            patch_size: int,
            hidden_dim: int,
            in_channels: int
        ):
        super().__init__()
        img_size, patch_size = make_tuple(img_size), make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) \
            * (img_size[1] // patch_size[1])

        self.pos_emb = torch.nn.Embedding(
            self.num_patches,
            hidden_dim
        )
        
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        P = self.conv(X).flatten(2).transpose(1, 2)
        _, l, _ = P.shape
        positions = (torch
            .arange(0, l)
            .expand(P.shape[0], -1)
            .to(P.device)
        )
        return P + self.pos_emb(positions)