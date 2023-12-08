from typing import List
import torch
from pytfex.transformer.node_router import RouteTensor


class Nodes(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_nodes: int,
            # dropout: float=0.5,
        ):
        super(Nodes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.nodes_1 = torch.nn.Parameter(
            torch.randn(
                (
                    self.num_nodes,
                    self.hidden_dim,
                    4 * self.hidden_dim
                )
            ),
            requires_grad=True
        )

        self.nodes_2 = torch.nn.Parameter(
            torch.randn(
                (
                    self.num_nodes,
                    4 * self.hidden_dim,
                    self.hidden_dim
                )
            ),
            requires_grad=True
        )

    def forward(
            self,
            x :RouteTensor,
            node_ind :RouteTensor
        ):
        pass
