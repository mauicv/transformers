from pytfex.model.heads import ClassificationHead, InversePatch
import torch

def test_inverse_patch():
    head = InversePatch(
        img_size=(28, 28),
        patch_size=(7, 7),
        hidden_dim=12,
        in_channels=1,
    )
    t1 = torch.randn((1, 16, 12))
    t2 = head(t1)
    assert t2.shape == (1, 16, 49)
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