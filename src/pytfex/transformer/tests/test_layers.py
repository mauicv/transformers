from pytfex.transformer.attention import Attention, RelativeAttention, GumbelSoftmaxRelativeAttention
from pytfex.transformer.mlp import MLP
from pytfex.transformer.moe_ec import ExpertChoiceMoE
from pytfex.transformer.moe_tc import TokenChoiceMoE
from pytfex.transformer.layer import TransformerLayer
import torch


def test_attention():
    attn = Attention(
        hidden_dim=12,
        num_heads=4,
        dropout=0.5
    )

    t1 = torch.ones((1, 10, 12))
    t2, _ = attn(t1)
    assert t2.shape == (1, 10, 12)


def test_attention_kv_cache():
    attn = Attention(
        hidden_dim=12,
        num_heads=4,
        dropout=0.0
    )

    t1 = torch.randn((2, 10, 12))
    _t2, kv_cache = attn(t1, use_kv_cache=True)
    kv_cache.q = [kv_cache.q[-1][:, :, :-1, :]]
    kv_cache.k = [kv_cache.k[-1][:, :, :-1, :]]
    kv_cache.v = [kv_cache.v[-1][:, :, :-1, :]]
    t1 = t1[:, [-1], :]
    t2, kv_cache = attn(t1, use_kv_cache=True, kv_cache=kv_cache)
    assert t2.shape == (2, 1, 12)
    for a, b in zip(
            t2[:, [-1]].flatten().tolist(),
            _t2[:, [-1]].flatten().tolist()
        ):
        assert abs(a - b) < 1e-6, f"{a} != {b}"
    assert kv_cache.k[0].shape == (2, 4, 9, 3)
    assert kv_cache.v[0].shape == (2, 4, 9, 3)
    assert kv_cache.k[1].shape == (2, 4, 1, 3)
    assert kv_cache.v[1].shape == (2, 4, 1, 3)
    assert len(kv_cache.q) == 2


def test_rel_attention():
    attn = RelativeAttention(
        hidden_dim=12,
        num_heads=4,
        num_positions=10,
        dropout=0.5
    )

    t1 = torch.ones((1, 10, 12))
    t2, _ = attn(t1)
    assert t2.shape == (1, 10, 12)


def test_rel_attention_kv_cache():
    attn = RelativeAttention(
        hidden_dim=12,
        num_heads=4,
        num_positions=10,
        dropout=0.0
    )

    t1 = torch.randn((2, 10, 12))
    _t2, kv_cache = attn(t1, use_kv_cache=True)
    t1 = t1[:, [-1], :]
    kv_cache.q = [kv_cache.q[-1][:, :, :-1, :]]
    kv_cache.k = [kv_cache.k[-1][:, :, :-1, :]]
    kv_cache.v = [kv_cache.v[-1][:, :, :-1, :]]
    t2, kv_cache = attn(t1, use_kv_cache=True, kv_cache=kv_cache)
    assert t2.shape == (2, 1, 12)
    for a, b in zip(
            t2[:, [-1]].flatten().tolist(),
            _t2[:, [-1]].flatten().tolist()
        ):
        assert abs(a - b) < 1e-6, f"{a} != {b}"
    assert kv_cache.k[0].shape == (2, 4, 9, 3)
    assert kv_cache.v[0].shape == (2, 4, 9, 3)
    assert kv_cache.k[1].shape == (2, 4, 1, 3)
    assert kv_cache.v[1].shape == (2, 4, 1, 3)
    assert len(kv_cache.q) == 2


def test_gumbel_softmax_rel_attention():
    attn = GumbelSoftmaxRelativeAttention(
        hidden_dim=12,
        num_heads=4,
        num_positions=10,
        dropout=0.5
    )

    t1 = torch.ones((1, 10, 12))
    t2, _ = attn(t1)
    assert t2.shape == (1, 10, 12)

    attn.set_tau(0.5)
    attn.set_hard(True)
    t3, _ = attn(t1)
    assert t3.shape == (1, 10, 12)


def test_gumbel_softmax_rel_attention_kv_cache():
    attn = GumbelSoftmaxRelativeAttention(
        hidden_dim=12,
        num_heads=4,
        num_positions=10,
        dropout=0.0
    )
    attn.set_tau(0.0)
    attn.set_hard(True)
    t1 = torch.randn((2, 10, 12))
    _t2, kv_cache = attn(t1, use_kv_cache=True)
    t1 = t1[:, [-1], :]
    kv_cache.q = [kv_cache.q[-1][:, :, :-1, :]]
    kv_cache.k = [kv_cache.k[-1][:, :, :-1, :]]
    kv_cache.v = [kv_cache.v[-1][:, :, :-1, :]]
    t2, kv_cache = attn(t1, use_kv_cache=True, kv_cache=kv_cache)
    assert t2.shape == (2, 1, 12)
    for a, b in zip(
            t2[:, [-1]].flatten().tolist(),
            _t2[:, [-1]].flatten().tolist()
        ):
        assert abs(a - b) < 1e-6, f"{a} != {b}"
    assert kv_cache.k[0].shape == (2, 4, 9, 3)
    assert kv_cache.v[0].shape == (2, 4, 9, 3)
    assert kv_cache.k[1].shape == (2, 4, 1, 3)
    assert kv_cache.v[1].shape == (2, 4, 1, 3)
    assert len(kv_cache.q) == 2


def test_MLP():
    mlp = MLP(
        hidden_dim=12,
        dropout=0.5
    )
    t1 = torch.ones((1, 10, 12))
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


def test_transformer_layer():
    layer = TransformerLayer(
        hidden_dim=12,
        attn=Attention(
            hidden_dim=12,
            num_heads=4,
            dropout=0.0
        ),
        mlp=MLP(
            hidden_dim=12,
            dropout=0.0
        )
    )
    t1 = torch.randn((1, 3, 12))
    t21, _ = layer(t1)
    t22, _ = layer(t1, use_kv_cache=True, kv_cache=None)
    for a, b in zip(
            t21[:, [-1]].flatten().tolist(),
            t22[:, [-1]].flatten().tolist()
        ):
        assert abs(a - b) < 1e-6, f"{a} != {b}"
    assert t21.shape == (1, 3, 12)
    assert t22.shape == (1, 3, 12)