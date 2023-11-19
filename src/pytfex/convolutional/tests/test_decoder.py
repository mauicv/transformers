import torch
from pytfex.convolutional.decoder import DecoderLayer, Decoder

def test_decoder_layer():
    enc = DecoderLayer(
        in_filters=32,
        out_filters=64,
        num_residual=2,
    )
    t1 = torch.randn(1, 32, 64, 64)
    t2 = enc(t1)
    assert t2.shape == (1, 64, 128, 128)


def test_decoder():
    layers = [
        DecoderLayer(
            in_filters=128,
            out_filters=64,
            num_residual=2,
        ),
        DecoderLayer(
            in_filters=64,
            out_filters=32,
            num_residual=2,
        ),
    ]
    enc = Decoder(
        nc=3,
        ndf=32,
        layers=layers,
    )

    t1 = torch.randn(1, 128, 32, 32)
    t2 = enc(t1)
    assert t2.shape == (1, 3, 128, 128)