import torch
import pytest
from pytfex.model.embedders import TokenPositionEmbedder, PatchEmbedder, PositionEmbedder


def test_token_position_embedder():
    embedder = TokenPositionEmbedder(
        max_sequence_length=100,
        dictionary_size=100,
        hidden_dim=12
    )
    t1 = torch.randint(100, (1, 10))
    t2 = embedder(t1)
    assert t2.shape == (1, 10, 12)


@pytest.mark.parametrize(
    "overlap,num_patches",
    [
        ((0, 0), 16),
        ((3, 3), 36)
    ])
def test_patch_embedder(overlap, num_patches):
    embedder = PatchEmbedder(
        img_size=(28, 28),
        patch_size=(7, 7),
        overlap=overlap,
        hidden_dim=12,
        in_channels=1
    )
    t1 = torch.randn((1, 1, 28, 28))
    t2 = embedder.get_patches(t1)
    assert embedder.num_patches == num_patches
    assert t2.shape == (1, num_patches, 7*7)
    t3 = embedder(t2)
    assert t3.shape == (1, num_patches, 12)


def test_position_embedder():
    embedder = PositionEmbedder(
        number_positions=10,
        hidden_dim=12
    )
    x = torch.randn((64, 1, 12))
    x = x.expand(-1, 5, -1)
    t = embedder(x)
    assert t.shape == (64, 5, 12)