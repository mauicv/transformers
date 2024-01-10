import torch


class SparseMLP(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            intermediate_dim: int=None,
            dropout: float=0.5,
            k: int=5,
        ):
        super(SparseMLP, self).__init__()
        self.hidden_dim = hidden_dim
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4
        self.intermediate_dim = intermediate_dim
        self.k = k

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
        self.embedding = torch.nn.Embedding(
            self.intermediate_dim,
            self.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        G, I = torch.topk(x, self.k, dim=-1)
        x = self.embedding(I) * G.unsqueeze(-1)
        x = x.sum(dim=-2)
        x = self.mlp_dropout(x)
        return x