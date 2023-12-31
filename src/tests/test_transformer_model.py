import pytest
from pytfex.transformer.gpt import GPT
from pytfex.transformer.make_model import init_from_file
from pytfex.transformer.mask import get_causal_mask
import torch

def test_model():
    model = init_from_file('./src/tests/test_yaml/simple.yml')
    t1 = torch.randint(1000, (1, 10))
    t2 = model(t1)
    assert t2.shape == (1, 10, 12)


def test_model_with_mask():
    model = init_from_file('./src/tests/test_yaml/simple.yml')
    t1 = torch.randint(1000, (3, 10))
    mask = get_causal_mask(10)
    t2 = model(t1, mask=mask)
    assert not torch.isnan(t2).any()
    assert t2.shape == (3, 10, 12)