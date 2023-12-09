import torch
from random import randint
import pytest

from pytfex.transformer.node_router import RouteTensor
from pytfex.transformer.node_array import RoutingModelLayer
from pytfex.transformer.routing_net import RoutingModel
from pytfex.transformer.heads import ClassificationHead
from pytfex.transformer.embedders import TokenEmbedder


@pytest.fixture
def model():
    layers = [
        RoutingModelLayer(
            hidden_dim=128,
            num_nodes=10,
            num_heads=5,
            k=5,
            dropout=0.1,
        ),
        RoutingModelLayer(
            hidden_dim=128,
            num_nodes=5,
            num_heads=5,
            k=2,
            dropout=0.1,
        ),
        RoutingModelLayer(
            hidden_dim=128,
            num_nodes=1,
            num_heads=1,
            k=1,
            dropout=0.1,
        ),
    ]

    model = RoutingModel(
        hidden_dim=128,
        layers=layers,
        dropout=0.1,
        embedder=TokenEmbedder(
            dictionary_size=10,
            hidden_dim=128
        ),
        head=ClassificationHead(
            hidden_dim=128,
            vocab_size=10
        )
    )
    return model


def test_RoutingModel(model):
    a = torch.randint(0, 10, (4, ))
    b = torch.randint(0, 10, (4, ))
    x = RouteTensor(data=[a, b])
    y = model(x)
    assert y.shapes == ((1, 10), (1, 10))


def test_RoutingModel_grad(model):
    opt = torch.optim.Adam(model.parameters())
    opt.zero_grad()
    a = torch.randint(0, 10, (4, ))
    b = torch.randint(0, 10, (4, ))
    x = RouteTensor(data=[a, b])
    y = model(x)
    lab = torch.randint(0, 10, (2,))
    cce = torch.nn.CrossEntropyLoss()
    loss = cce(y.data, lab)
    loss.backward()
    opt.step()
