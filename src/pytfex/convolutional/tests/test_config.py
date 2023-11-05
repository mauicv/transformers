from pytfex.convolutional.make_model import init_from_yml_string
import torch


def test_encoder_model_config():
    config = """
    type: 'Encoder'
    params:
        nc: 3
        ndf: 32
        layers:
        -   type: 'EncoderLayer'
            params:
                in_channels: 32
                out_channels: 64
                num_residual: 2
        -   type: 'EncoderLayer'
            params:
                in_channels: 64
                out_channels: 128
                num_residual: 1
        output_layer:
            type: 'ConvolutionalLayer'
            params:
                in_channels: 128
                out_channels: 4
                activation: 'identity'
    """
    model = init_from_yml_string(config)
    t1 = torch.randn(1, 3, 128, 128)
    t2 = model(t1)
    assert t2.shape == (1, 4, 32, 32)


def test_decoder_model_config():
    config = """
    type: 'Decoder'
    params:
        nc: 3
        ndf: 32
        output_activation: 
            type: 'Tanh'
            params: {}
        layers:
        -   type: 'DecoderLayer'
            params:
                in_filters: 128
                out_filters: 64
                num_residual: 2
        -   type: 'DecoderLayer'
            params:
                in_filters: 64
                out_filters: 32
                num_residual: 2
        input_layer:
            type: 'ConvolutionalLayer'
            params:
                in_channels: 4
                out_channels: 128
                activation: 'identity'
    """
    model = init_from_yml_string(config)
    t1 = torch.randn(1, 4,  32, 32)
    t2 = model(t1)
    assert t2.shape == (1, 3, 128, 128)
    assert torch.all(t2 >= -1.0)
    assert torch.all(t2 <= 1.0)