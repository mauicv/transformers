from pytfex.transformer.base import BaseTransformer
import torch.nn as nn
import torch
from typing import Optional
from pytfex.transformer.experimental.soup import RelativeAttentionSoup, ExpertChoiceSoup


class WeeSoupyBoy(torch.nn.Module, BaseTransformer):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            blk_size: Optional[int] = None,
            dropout: float=0.5,
            embedder: torch.nn.Module=None,
            attention_soup: RelativeAttentionSoup=None,
            mlp_soup: ExpertChoiceSoup=None,
            head: torch.nn.Module=None,
            depth: int=2,
        ):
        super(WeeSoupyBoy, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.blk_size = blk_size

        self.drop = nn.Dropout(dropout)
        self.embedder = embedder
        self.attention_soup = attention_soup
        self.mlp_soup = mlp_soup
        self.head = head
        self.depth = depth

    def forward(self, x, mask=None):
        if self.embedder:
            x = self.embedder(x)

        x = self.drop(x)
        for i in range(self.depth):
            x = self.attention_soup(x, mask=mask)
            x = self.mlp_soup(x)
        if self.head is not None:
            x = self.head(x)
        return x
