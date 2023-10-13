from src.model.make import init_from_config
import yaml
import os

def test_model():
    with open('./tests/test_yaml/config.yml', 'r') as file:
        config = yaml.safe_load(file)
    model = init_from_config(config)
    assert model.hidden_dim == 12
    assert len(model.layers) == 2 + 5
    n_heads_list = [layer.attn.num_heads for layer in model.layers]
    assert n_heads_list == [4, 4, 3, 3, 3, 3, 3]