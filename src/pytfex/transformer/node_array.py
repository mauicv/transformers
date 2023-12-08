from typing import List
import torch
from pytfex.transformer.node_router import RouteTensor


class LinearNodes(torch.nn.Module):
    def __init__(
            self,
            num_nodes: int,
            input_dim: int,
            output_dim: int,
            bias=True,
        ):
        super(LinearNodes, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes

        self.weight = torch.nn.Parameter(
            torch.randn(
                (
                    self.num_nodes,
                    output_dim,
                    input_dim,
                )
            ),
            requires_grad=True
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.randn(
                    (
                        self.num_nodes,
                        output_dim
                    )
                ),
                requires_grad=True
            )

    def forward(
            self,
            x :RouteTensor,
            node_ind :RouteTensor
        ) -> RouteTensor:
        weights = self.weight[node_ind.data]
        biases = self.bias[node_ind.data]
        data = x.data[:, None, :] @ weights.transpose(-1, -2)
        data = data.squeeze(1) + biases
        return RouteTensor(
            data=data,
            batch_inds=x.batch_inds
        )


class MLPNodes(torch.nn.Module):
    def __init__(
            self,
            num_nodes: int,
            hidden_dim: int,
            dropout: float=0.5,
        ):
        super(MLPNodes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.mlp_dropout = torch.nn.Dropout(
            torch.tensor(
                dropout,
                dtype=torch.float32
            )
        )

        self.linear1 = LinearNodes(
            num_nodes=num_nodes,
            input_dim=hidden_dim,
            output_dim=4 * hidden_dim
        )

        self.linear2 = LinearNodes(
            num_nodes=num_nodes,
            input_dim=4 * hidden_dim,
            output_dim=hidden_dim
        )

    def forward(
            self,
            x: RouteTensor,
            node_ind: RouteTensor
        ) -> RouteTensor:
        x = self.linear1(x, node_ind)
        x = x.apply(torch.nn.functional.gelu)
        x = self.linear2(x, node_ind)
        x = x.apply(self.mlp_dropout)
        return x

