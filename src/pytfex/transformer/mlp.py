import torch


class MLP(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            intermediate_dim: int=None,
            dropout: float=0.5,
        ):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4
        self.intermediate_dim = intermediate_dim

        self.dropout = torch.tensor(
            dropout,
            dtype=torch.float32
        )
        self.mlp_dropout = torch.nn.Dropout(
            self.dropout
        )

        self.linear1 = torch.nn.Linear(
            self.hidden_dim,
            self.intermediate_dim
        )
        self.linear2 = torch.nn.Linear(
            self.intermediate_dim,
            self.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        x = self.mlp_dropout(x)
        return x