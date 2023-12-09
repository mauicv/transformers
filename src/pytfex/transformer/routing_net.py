from pytfex.transformer.base import BaseTransformer
from pytfex.transformer.node_router import RouteTensor
from pytfex.transformer.node_array import RoutingModelLayer
import torch.nn as nn
import torch


class RoutingModel(torch.nn.Module, BaseTransformer):
    def __init__(
            self,
            hidden_dim: int,
            embedder: torch.nn.Module=None,
            layers: list[RoutingModelLayer] = [],
            head: torch.nn.Module=None,
            dropout: float=0.5,
        ):
        super(RoutingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.embedder = embedder
        self.head = head
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedder:
            x = self.embedder(x)
        x = x.apply(self.drop)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f'layer-{i}, {x.data.min()}, {x.data.max()}')
        if self.head:
            x = x.apply(self.head)
        return x
