import torch

class Attention(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float=0.5,
        ) -> None:
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = torch.tensor(
            dropout,
            dtype=torch.float32
        )
        self.attn_dropout = torch.nn.Dropout(self.dropout)
        self.resid_dropout = torch.nn.Dropout(self.dropout)

        self.qkv = torch.nn.Linear(
            self.hidden_dim,
            3 * self.hidden_dim
        )
        self.linear = torch.nn.Linear(
            self.hidden_dim,
            self.hidden_dim
        )

        self.head_dim = self.hidden_dim // self.num_heads

    def forward(
                self,
                x,
                mask=None
            ):
        if mask is not None:
            assert mask.dtype == torch.bool, "Mask must be of type torch.float32"
        assert len(x.shape) == 3, "Input must have shape (batch, seq_len, hidden_dim)"
        assert x.shape[-1] == self.hidden_dim, "Input has incorrect hidden_dim"
        b, l, d = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, 3, self.num_heads, l, -1)
        q, k, v = torch.split(qkv, [1, 1, 1], dim=1)
        q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)
        a = k @ q.transpose(-2, -1)
        if mask is not None:
            a = a.masked_fill(mask == 0, float('-inf'))
        a = self.attn_dropout(a)
        hd = torch.tensor(self.head_dim, dtype=torch.float32)
        a = torch.softmax(a / torch.sqrt(hd), dim=-1)
        output =  (a @ v).transpose(1, 2).reshape(b, l, d)
        output = self.linear(output)
        return self.resid_dropout(output)
