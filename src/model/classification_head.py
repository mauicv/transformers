import torch


class ClassificationHead(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            vocab_size: int,
        ):
        super(ClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.ln_f = torch.nn.LayerNorm(
            self.hidden_dim
        )
        self.linear = torch.nn.Linear(
            self.hidden_dim,
            self.vocab_size
        )

    def forward(self, x):
        return self.linear(self.ln_f(x))
