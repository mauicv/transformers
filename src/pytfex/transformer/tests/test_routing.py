import torch
from random import randint
import pytest
from pytfex.transformer.node_router import NodeRouter, RouteTensor
from pytfex.transformer.node_array import LinearNodes, MLPNodes, RoutingModelLayer


def test_RouteTensor():
    batch = [torch.randn((randint(2, 6), 128)) for _ in range(4)]
    x = RouteTensor(data=batch)
    assert sum([s[0] for s in x.shapes]) == x.shape[0]
    L = torch.nn.Linear(128, 128)
    y = x.apply(L)
    assert x.shapes == y.shapes
    assert x.shape == y.shape


def test_NodeRouter__match_gates():
    router = NodeRouter(
        hidden_dim=128,
        num_nodes=5,
        num_heads=2,
        k=3,
    )
    a = torch.randn((4, 128))
    b = torch.randn((4, 128))
    x = RouteTensor(data=[a, b])
    y, node_i, head_i = router._match_gates(x)
    # y.shape -> ((len(x), head, hidden_dim), ...)
    assert y.shapes == ((4, 3, 128), (4, 3, 128))
    assert node_i.shapes == ((4, 3), (4, 3))
    assert head_i.shapes == ((4, 3), (4, 3))


@pytest.mark.parametrize('num_nodes', [
    5,
    25
])
def test_NodeRouter_forward(num_nodes):
    router = NodeRouter(
        hidden_dim=128,
        num_nodes=num_nodes,
        num_heads=2,
        k=3
    )
    a = torch.randn((4, 128))
    b = torch.randn((4, 128))
    x = RouteTensor(data=[a, b])
    z, node_i = router(x)
    for a, b in zip(z.shapes, node_i.shapes):
        assert a[0] == b[0]


@pytest.mark.parametrize('num_nodes', [
    5,
    25
])
def test_Nodes_forward(num_nodes):
    router = NodeRouter(
        hidden_dim=128,
        num_nodes=num_nodes,
        num_heads=2,
        k=3
    )
    nodes = LinearNodes(
        num_nodes=num_nodes,
        output_dim=4*128,
        input_dim=128,
    )
    a = torch.randn((4, 128))
    b = torch.randn((4, 128))
    x = RouteTensor(data=[a, b])
    z, node_i = router(x)
    y = nodes(z, node_i)

    for a, b in zip(z.shapes, y.shapes):
        assert a[0] == b[0]
        assert a[1] * 4 == b[1]
    

@pytest.mark.parametrize('num_nodes', [
    5,
    25
])
def test_MLPNodes_forward(num_nodes):
    router = NodeRouter(
        hidden_dim=128,
        num_nodes=num_nodes,
        num_heads=2,
        k=3
    )
    nodes = MLPNodes(
        num_nodes=num_nodes,
        hidden_dim=128,
        dropout=0.5,
    )
    a = torch.randn((4, 128))
    b = torch.randn((4, 128))
    x = RouteTensor(data=[a, b])
    z, node_i = router(x)
    y = nodes(z, node_i)

    for a, b in zip(z.shapes, y.shapes):
        assert a[0] == b[0]
        assert a[1] == b[1]


@pytest.mark.parametrize('num_nodes', [
    5,
    25
])
def test_RoutingModelLayer_forward(num_nodes):
    layer = RoutingModelLayer(
        hidden_dim=128,
        num_nodes=num_nodes,
        num_heads=2,
        k=3,
        dropout=0.5,
    )
    a = torch.randn((4, 128))
    b = torch.randn((4, 128))
    x = RouteTensor(data=[a, b])
    y = layer(x)
    assert len(x.shapes) == len(y.shapes)