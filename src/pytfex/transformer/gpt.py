from pytfex.transformer.base import BaseTransformer
import torch.nn as nn
import torch
from typing import Optional
from pytfex.transformer.attention import LayerKVQCache

class KVCache:
    def __init__(self, batch_size: int, head_dim: int, num_heads: int, max_len: int, num_layers: int):
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.layers = [LayerKVQCache() for _ in range(num_layers)]

    def size(self):
        return self.layers[-1].size()
    
    def __len__(self):
        return len(self.layers)


class GPT(torch.nn.Module, BaseTransformer):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            blk_size: Optional[int] = None,
            dropout: float=0.5,
            embedder: torch.nn.Module=None,
            layers: list[torch.nn.Module] = [],
            head: torch.nn.Module=None,
        ):
        super(GPT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.blk_size = blk_size

        self.drop = nn.Dropout(dropout)
        self.embedder = embedder
        self.layers = torch.nn.ModuleList(layers)
        self.head = head

    def forward(self, x, mask=None, use_kv_cache=False, kv_cache=None):
        if self.embedder:
            x = self.embedder(x, kv_cache=kv_cache)

        if kv_cache is None and use_kv_cache:
            kv_cache = KVCache(
                batch_size=x.shape[0],
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                max_len=self.blk_size,
                num_layers=len(self.layers)
            )
        
        x = self.drop(x)
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache.layers[layer_idx] if use_kv_cache else None
            x, _ = layer(
                x,
                mask=mask,
                use_kv_cache=use_kv_cache,
                kv_cache=layer_cache
            )
        if self.head is not None:
            x = self.head(x)
        if not use_kv_cache:
            return x
        return x, kv_cache
