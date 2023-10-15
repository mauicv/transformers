import pytest
from pytfex.model.gpt import GPT
from pytfex.model.make_model import init_from_file
import torch


def test_model_errors():
    with pytest.raises(AssertionError):
        GPT(
            hidden_dim=12,
            num_heads=5,
            dropout=0.5
        )

def test_model():
    model = init_from_file('./src/tests/test_yaml/simple.yml')
    t1 = torch.randint(1000, (1, 10))
    t2 = model(t1)
    assert t2.shape == (1, 10, 12)