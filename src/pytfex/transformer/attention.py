import torch

class Attention(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float=0.5,
        ) -> None:
        super(Attention, self).__init__()
        assert hidden_dim % num_heads == 0, f"num_heads must divide hidden_dim, {hidden_dim=}, {num_heads=}"
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
        q, k, v = torch.split(qkv, self.hidden_dim, dim=2)
        q = q.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        hd = torch.tensor(self.head_dim, dtype=torch.float32)
        a = q @ k.transpose(-2, -1) / torch.sqrt(hd)
        if mask is not None:
            a = a.masked_fill(mask, float('-inf'))
        a = torch.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        output =  (a @ v).transpose(1, 2).reshape(b, l, d)
        output = self.linear(output)
        return self.resid_dropout(output)
