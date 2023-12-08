from pytfex.transformer.base import BaseTransformer
import torch.nn as nn
import torch


class RoutingNet(torch.nn.Module, BaseTransformer):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float=0.5,
            embedder: torch.nn.Module=None,
            nodes: list[torch.nn.Module] = [],
        ):
        super(RoutingNet, self).__init__()
        assert hidden_dim % num_heads == 0, "num_heads must divide hidden_dim"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.embedder = embedder
        self.nodes = torch.nn.ModuleList(nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedder:
            x = self.embedder(x)
        x = self.drop(x)
        # TODO: Magic!
        return x
