from pytfex.transformer.make_model import init_from_config, TransformerObjectRegistry, \
    init_from_yml_string
import torch
import yaml
import os

def test_model_config():
    with open('./src/tests/test_yaml/config.yml', 'r') as file:
        config = yaml.safe_load(file)
    model = init_from_config(config)
    assert model.hidden_dim == 12
    assert len(model.layers) == 2 + 5
    n_heads_list = [layer.attn.num_heads for layer in model.layers]
    assert n_heads_list == [4, 4, 3, 3, 3, 3, 3]
    assert model.layers[0].ln_1 != model.layers[1].ln_1
    assert model.layers[0].ln_2 != model.layers[1].ln_2
    assert model.layers[0].attn != model.layers[1].attn
    assert model.layers[0].mlp != model.layers[1].mlp


def test_model_config_load_state(tmpdir):
    with open('./src/tests/test_yaml/simple.yml', 'r') as file:
        config = yaml.safe_load(file)
    model_1 = init_from_config(config)
    model_1.save_state(os.path.join(tmpdir, 'model_state.pt'))

    with open('./src/tests/test_yaml/simple.yml', 'r') as file:
        config = yaml.safe_load(file)
    model_2 = init_from_config(config, load_state=True, path=tmpdir)

    for layer_1, layer_2 in zip(model_1.layers, model_2.layers):
        assert torch.allclose(layer_1.attn.qkv.weight, layer_2.attn.qkv.weight)


def test_custom_module_init():
    @TransformerObjectRegistry.register('CustomModule')
    class CustomModule(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            self.s = s

        def forward(self, x):
            return x * self.s

    model = init_from_yml_string("""
    type: 'CustomModule'
    params:
        s: 2
    """)

    assert model(torch.tensor([1.])) == torch.tensor([2.])