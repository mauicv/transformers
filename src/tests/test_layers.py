from pytfex.model.attention import Attention
from pytfex.model.mlp import MLP
import torch


def test_attention():
    attn = Attention(
        hidden_dim=12,
        num_heads=4,
        dropout=0.5
    )

    t1 = torch.zeros((1, 10, 12))
    t2 = attn(t1)
    assert t2.shape == (1, 10, 12)


def test_MLP():
    mlp = MLP(
        hidden_dim=12,
        dropout=0.5
    )
    t1 = torch.zeros((1, 10, 12))
    t2 = mlp(t1)
    assert t2.shape == (1, 10, 12)