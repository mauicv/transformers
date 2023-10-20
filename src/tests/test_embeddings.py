import torch
from pytfex.model.embedders import TokenPositionEmbedder, PatchEmbedder


def test_token_position_embedder():
    embedder = TokenPositionEmbedder(
        max_sequence_length=100,
        dictionary_size=100,
        hidden_dim=12
    )
    t1 = torch.randint(100, (1, 10))
    t2 = embedder(t1)
    assert t2.shape == (1, 10, 12)


def test_patch_embedder():
    embedder = PatchEmbedder(
        img_size=(28, 28),
        patch_size=(7, 7),
        hidden_dim=12,
        in_channels=1
    )
    t1 = torch.randn((1, 1, 28, 28))
    t2 = embedder(t1)
    assert t2.shape == (1, 16, 12)
