import torch


class MLP(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            dropout: float=0.5,
        ):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = torch.tensor(
            dropout,
            dtype=torch.float32
        )
        self.mlp_dropout = torch.nn.Dropout(
            self.dropout
        )

        self.linear1 = torch.nn.Linear(
            self.hidden_dim,
            4 * self.hidden_dim
        )
        self.linear2 = torch.nn.Linear(
            4 * self.hidden_dim,
            self.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.mlp_dropout(x)
        x = self.linear2(x)
        return x