from pytfex.transformer.attention import Attention
from pytfex.transformer.mlp import MLP
from pytfex.transformer.moe import MoE
from pytfex.transformer.mof import MoF, MoF2
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


def test_MoE_MLP():
    mlp = MoE(
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


def test_MoF_MLP():
    mod = MoF(
        hidden_dim=12,
        model=MLP(
            hidden_dim=8,
            dropout=0.5
        ),
        num_groups=3,
        k=2
    )
    t1 = torch.randn((2, 10, 12))
    t2 = mod(t1)
    assert t2.shape == (2, 10, 12)


def test_MoE_MoF_MLP():
    mlp = MoE(
        hidden_dim=12,
        c=2,
        experts=[
            MoF(
                hidden_dim=12,
                model=MLP(
                    hidden_dim=8,
                    dropout=0.5
                ),
                num_groups=3,
                k=2
            ) for _ in range(4)
        ]
    )
    t1 = torch.randn((2, 10, 12))
    t2 = mlp(t1)
    assert t2.shape == (2, 10, 12)


def test_MOF_attention():
    mof_attn = MoF(
        hidden_dim=12,
        model=Attention(
            hidden_dim=8,
            num_heads=4,
            dropout=0.5
        ),
        num_groups=3,
        k=2
    )

    t1 = torch.randn((2, 10, 12))
    t2 = mof_attn(t1)
    assert t2.shape == (2, 10, 12)


def test_MoF2_MLP():
    mod = MoF2(
        hidden_dim=12,
        model=MLP(
            hidden_dim=4,
            dropout=0.5
        ),
        num_proj=3,
        k=2
    )
    t1 = torch.randn((2, 10, 12))
    t2 = mod(t1)
    assert t2.shape == (2, 10, 12)


def test_MOF2_attention():
    mof_attn = MoF2(
        hidden_dim=12,
        model=Attention(
            hidden_dim=4,
            num_heads=4,
            dropout=0.5
        ),
        num_proj=3,
        k=2
    )

    t1 = torch.randn((2, 10, 12))
    t2 = mof_attn(t1)
    assert t2.shape == (2, 10, 12)