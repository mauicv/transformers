from pytfex.transformer.base import BaseTransformer
import torch.nn as nn
import torch


class GPT(torch.nn.Module, BaseTransformer):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
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

        self.drop = nn.Dropout(dropout)
        self.embedder = embedder
        self.layers = torch.nn.ModuleList(layers)
        self.head = head

    def _init_kv_cache(self):
        return [
            {
                'k': torch.zeros(1, self.num_heads, 0, self.head_dim),
                'v': torch.zeros(1, self.num_heads, 0, self.head_dim)
            } for _ in range(len(self.layers))
        ]

    def forward(self, x, mask=None, use_kv_cache=False, kv_cache=None):
        if use_kv_cache and kv_cache is not None:
            assert len(kv_cache) == len(self.layers), \
                'kv_cache must be a list of dicts with the same length as layers'
        if kv_cache is None:
            kv_cache = self._init_kv_cache()

        if self.embedder:
            x = self.embedder(x, kv_cache=kv_cache)
        x = self.drop(x)
        new_kv_cache = []
        for layer, kv_cache in zip(self.layers, kv_cache):
            x, _kv_cache = layer(
                x,
                mask=mask,
                use_kv_cache=use_kv_cache,
                kv_cache=kv_cache
            )
            new_kv_cache.append(_kv_cache)
        if self.head is not None:
            x = self.head(x)
        if not use_kv_cache:
            return x
        return x, new_kv_cache
