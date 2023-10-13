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
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x