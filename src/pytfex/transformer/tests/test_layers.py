from pytfex.transformer.attention import Attention, RelativeAttention
from pytfex.transformer.mlp import MLP
from pytfex.transformer.moe_ec import ExpertChoiceMoE
from pytfex.transformer.moe_tc import TokenChoiceMoE
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


def test_rel_attention():
    attn = RelativeAttention(
        hidden_dim=12,
        num_heads=4,
        num_positions=10,
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


def test_expert_choice_MoE_MLP():
    mlp = ExpertChoiceMoE(
        hidden_dim=12,
        c=2,
        experts=[
            MLP(
                hidden_dim=12,
                dropout=0.5
            ) for _ in range(4)
        ]
    )
    t1 = torch.randn((2, 10, 12))
    t2 = mlp(t1)
    assert t2.shape == (2, 10, 12)


def test_token_choice_MoE_MLP():
    mlp = TokenChoiceMoE(
        hidden_dim=12,
        k=2,
        experts=[
            MLP(
                hidden_dim=12,
                dropout=0.5
            ) for _ in range(4)
        ]
    )
    t1 = torch.randn((2, 10, 12))
    t2 = mlp(t1)
    assert t2.shape == (2, 10, 12)
