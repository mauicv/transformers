import torch
from pytfex.convolutional.encoder import EncoderLayer, Encoder

def test_encoder_layer():
    enc = EncoderLayer(
        in_channels=32,
        out_channels=64,
        num_residual=2,
    )
    t1 = torch.randn(1, 32, 128, 128)
    t2 = enc(t1)
    assert t2.shape == (1, 64, 64, 64)


def test_encoder():
    layers = [
        EncoderLayer(
            in_channels=32,
            out_channels=64,
            num_residual=2,
        ),
        EncoderLayer(
            in_channels=64,
            out_channels=128,
            num_residual=2,
        ),
    ]
    enc = Encoder(
        nc=3,
        ndf=32,
        layers=layers,
    )

    t1 = torch.randn(1, 3, 128, 128)
    t2 = enc(t1)
    assert t2.shape == (1, 128, 32, 32)