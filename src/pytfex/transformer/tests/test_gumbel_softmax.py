from pytfex.transformer.gumbel_softmax import gumbel_softmax
import torch


def test_gumbel_softmax():
    logits = torch.randn(10, 10)
    tau = 1.0
    hard = True
    y_soft = gumbel_softmax(logits, tau, hard=hard)
    assert y_soft.shape == (10, 10)


def test_gumbel_softmax_hard():
    logits = torch.randn(10, 10)
    tau = 1.0
    hard = False
    y_hard = gumbel_softmax(logits, tau, hard=hard)
    assert y_hard.shape == (10, 10)