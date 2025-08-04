import torch
from pytfex.transformer.gumbel_softmax import gumbel_softmax


def _verify_input(x, mask, use_kv_cache, kv_cache, hidden_dim, head_dim):
    # if use_kv_cache and kv_cache is None:
    #     raise ValueError("kv_cache must be provided if use_kv_cache is True")

    # if use_kv_cache:
    #     if 'k' in kv_cache and 'v' in kv_cache:
    #         past_k, past_v = kv_cache['k'], kv_cache['v']
    #         assert past_k.shape[3] == head_dim, \
    #             f"kv_cache has incorrect head_dim, expected {head_dim}, got {past_k.shape[3]}"
    #         assert past_v.shape[3] == head_dim, \
    #             f"kv_cache has incorrect head_dim, expected {head_dim}, got {past_v.shape[3]}"
    if mask is not None:
        assert mask.dtype == torch.bool, "Mask must be of type torch.float32"
    assert len(x.shape) == 3, "Input must have shape (batch, seq_len, hidden_dim)"
    assert x.shape[-1] == hidden_dim, "Input has incorrect hidden_dim"


class LayerKVQCache:
    def __init__(self):
        self.k = []
        self.v = []
        self.q = []

    def write(self, k, v, q):
        self.k.append(k)
        self.v.append(v)
        self.q.append(q)

    def read(self):
        return (
            torch.cat(self.k, dim=2),
            torch.cat(self.v, dim=2),
            torch.cat(self.q, dim=2)
        )

    def size(self):
        return sum(k.shape[2] for k in self.k)


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
                mask=None,
                use_kv_cache=False,
                kv_cache=None,
            ):
        _verify_input(x, mask, use_kv_cache, kv_cache, self.hidden_dim, self.head_dim)
        b, l, d = x.shape

        if use_kv_cache and kv_cache is None:
            kv_cache = LayerKVQCache()

        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=2)
        q = q.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if use_kv_cache:
            kv_cache.write(k, v, q)
            k, v, _ = kv_cache.read()
            
        hd = torch.tensor(self.head_dim, dtype=torch.float32)
        a = q @ k.transpose(-2, -1) / torch.sqrt(hd)
        if mask is not None:
            a = a.masked_fill(mask, float('-inf'))
        a = torch.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        output = (a @ v).transpose(1, 2).reshape(b, l, d)
        output = self.linear(output)
        output = self.resid_dropout(output)
        return output, kv_cache


def generate_relative_positions(L):
    positions = torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
    return positions


class RelativeAttention(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_positions: int,
            dropout=0.1
        ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
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
        self.attn_drop = torch.nn.Dropout(0.1)
        self.resid_drop = torch.nn.Dropout(0.1)

        self.num_positions = num_positions
        self.Er = torch.nn.Parameter(torch.randn(
            self.num_heads, 
            self.num_positions,
            self.head_dim
        ))

    def forward(
            self,
            x,
            mask=None,
            use_kv_cache=False,
            kv_cache=None,
        ):
        _verify_input(x, mask, use_kv_cache, kv_cache, self.hidden_dim, self.head_dim)
        b, l, d = x.shape
        if use_kv_cache and kv_cache is None:
            kv_cache = LayerKVQCache()

        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=2)
        q = q.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if use_kv_cache:
            kv_cache.write(k, v, q)
            k, v, q = kv_cache.read()

        b, _, kv_l, _ = v.shape
        rel_pos = generate_relative_positions(kv_l)
        Er = self.Er[:, rel_pos].unsqueeze(0)
        # compute attention
        hd = torch.tensor(self.head_dim, dtype=torch.float32)
        a = q @ k.transpose(-2, -1) / torch.sqrt(hd)
        QEr = torch.einsum('bnlh,rnlkh->bnlk', q, Er)
        a = a + QEr

        if use_kv_cache:
            a = a[:, :, -l:, :]

        if mask is not None:
            a = a.masked_fill(mask, float('-inf'))
        a = torch.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        output =  (a @ v).transpose(1, 2).reshape(b, l, d)
        output = self.linear(output)
        output = self.resid_dropout(output)
        return output, kv_cache
    

class GumbelSoftmaxRelativeAttention(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_positions: int,
            dropout=0.1,
        ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
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
        self.attn_drop = torch.nn.Dropout(0.1)
        self.resid_drop = torch.nn.Dropout(0.1)

        self.num_positions = num_positions
        self.Er = torch.nn.Parameter(torch.randn(
            self.num_heads, 
            self.num_positions,
            self.head_dim
        ))

        self.tau=1.0
        self.hard=False

    def set_tau(self, tau):
        self.tau = tau
    
    def set_hard(self, hard):
        self.hard = hard

    def forward(
            self,
            x,
            mask=None,
            use_kv_cache=False,
            kv_cache=None,
        ):
        _verify_input(x, mask, use_kv_cache, kv_cache, self.hidden_dim, self.head_dim)
        b, l, d = x.shape

        if use_kv_cache and kv_cache is None:
            kv_cache = LayerKVQCache()

        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=2)
        q = q.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if use_kv_cache:
            kv_cache.write(k, v, q)
            k, v, q = kv_cache.read()

        b, _, kv_l, _ = v.shape
        rel_pos = generate_relative_positions(kv_l)
        Er = self.Er[:, rel_pos].unsqueeze(0)

        # compute attention
        hd = torch.tensor(self.head_dim, dtype=torch.float32)
        a = q @ k.transpose(-2, -1) / torch.sqrt(hd)
        QEr = torch.einsum('bnlh,rnlkh->bnlk', q, Er)
        a = a + QEr
        if use_kv_cache:
            a = a[:, :, -l:, :]

        if mask is not None:
            a = a.masked_fill(mask, float('-inf'))

        a = gumbel_softmax(a, self.tau, hard=self.hard)
        a = self.attn_dropout(a)
        output =  (a @ v).transpose(1, 2).reshape(b, l, d)
        output = self.linear(output)
        output = self.resid_dropout(output)
        return output, kv_cache
