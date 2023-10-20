import torch


class TokenPositionEmbedder(torch.nn.Module):
    def __init__(
            self, 
            max_sequence_length: int=None,
            dictionary_size: int=None,
            hidden_dim: int=None,
        ):
        super(TokenPositionEmbedder, self).__init__()

        self.pos_emb = torch.nn.Embedding(
            max_sequence_length,
            hidden_dim
        )

        self.tok_emb = torch.nn.Embedding(
            dictionary_size,
            hidden_dim
        )

    def forward(self, x):
        positions = (torch
            .arange(0, x.shape[1])
            .expand(x.shape[0], -1)
            .to(x.device)
        )
        x = self.tok_emb(x) + self.pos_emb(positions)
        return x
