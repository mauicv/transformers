import torch


class TransformerLayer(torch.nn.Module):
    def __init__(
            self,
            hidden_dim=None,
            attn=None,
            mlp=None,
        ):
        super(TransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = attn
        self.mlp = mlp
        self.ln_1 = torch.nn.LayerNorm(
            self.hidden_dim
        )
        self.ln_2 = torch.nn.LayerNorm(
            self.hidden_dim
        )

    def forward(self, x):
        _x = self.ln_1(x)
        x = x + self.attn(_x)
        _x = self.ln_2(x)
        x = x + self.mlp(_x)
        return x