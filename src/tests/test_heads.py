from pytfex.model.heads import ClassificationHead, InversePatch
import pytest
import torch


@pytest.mark.parametrize(
    "overlap,num_patches",
    [
        ((0, 0), 16),
        ((3, 3), 36)
    ])
def test_inverse_patch(overlap, num_patches):
    head = InversePatch(
        img_size=(28, 28),
        patch_size=(7, 7),
        overlap=overlap,
        hidden_dim=12,
        in_channels=1,
    )
    t1 = torch.randn((1, num_patches, 12))
    t2 = head(t1)
    assert t2.shape == (1, num_patches, 49)
    t2 = head.get_images(t2)
    assert t2.shape == (1, 1, 28, 28)

def test_classification_head():
    head = ClassificationHead(
        hidden_dim=12,
        vocab_size=100
    )
    t1 = torch.randn((1, 10, 12))
    t2 = head(t1)
    assert t2.shape == (1, 10, 100)