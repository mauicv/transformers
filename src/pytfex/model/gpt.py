from pytfex.model.base import BaseTransformer
import torch.nn as nn
import torch


class GPT(torch.nn.Module, BaseTransformer):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float=0.5,
            embedder: torch.nn.Module=None,
            layers: list[torch.nn.Module] = [],
            head: torch.nn.Module=None,
        ):
        super(GPT, self).__init__()
        assert hidden_dim % num_heads == 0, "num_heads must divide hidden_dim"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.embedder = embedder
        self.layers = torch.nn.ModuleList(layers)
        self.head = head

    def forward(self, x, mask=None):
        if self.embedder:
            x = self.embedder(x)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.head is not None:
            x = self.head(x)
        return x
