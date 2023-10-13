import torch
from src.model.layer import TransformerLayer


class Transformer(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float=0.5,
            max_sequence_length: int=1000,
            dictionary_size: int=1000,
            layers: list = []
        ):
        super(Transformer, self).__init__()
        assert hidden_dim % num_heads == 0, "num_heads must divide hidden_dim"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.dictionary_size = dictionary_size

        self.pos_emb = torch.nn.Embedding(
            self.max_sequence_length,
            self.hidden_dim
        )

        self.tok_emb = torch.nn.Embedding(
            self.dictionary_size,
            self.hidden_dim
        )

        print(layers)

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        positions = torch.arange(0, x.shape[1]).expand(x.shape[0], -1).to(x.device)
        x = self.tok_emb(x) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        return x