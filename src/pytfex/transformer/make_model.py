import yaml
import os
import copy

from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.attention import Attention, RelativeAttention
from pytfex.transformer.mlp import MLP
from pytfex.transformer.moe_ec import ExpertChoiceMoE
from pytfex.transformer.moe_tc import TokenChoiceMoE
from pytfex.transformer.gpt import GPT
from pytfex.transformer.heads import ClassificationHead, InversePatch
from pytfex.transformer.embedders import TokenEmbedder, PositionEmbedder, \
    MultiEmbedder, PatchEmbedder, LinearEmbedder


class TransformerObjectRegistry:
    _registry = {
        'TransformerLayer': TransformerLayer,
        'Attention': Attention,
        'MLP': MLP,
        'ExpertChoiceMoE': ExpertChoiceMoE,
        'TokenChoiceMoE': TokenChoiceMoE,
        'GPT': GPT,
        'ClassificationHead': ClassificationHead,
        'InversePatch': InversePatch,
        'TokenEmbedder': TokenEmbedder,
        'PositionEmbedder': PositionEmbedder,
        'MultiEmbedder': MultiEmbedder,
        'PatchEmbedder': PatchEmbedder,
        'LinearEmbedder': LinearEmbedder,
        'RelativeAttention': RelativeAttention
    }

    def register(name):
        def decorator(cls):
            TransformerObjectRegistry._registry[name] = cls
            return cls
        return decorator
    
    def get(name):
        return TransformerObjectRegistry._registry[name]


# TODO: udpate this for model state loading in init_from_config
def init_from_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return init_from_config(config)

# TODO: udpate this for model state loading in init_from_config
def init_from_yml_string(config_string):
    config = yaml.safe_load(config_string)
    return init_from_config(config)

def init_from_config(
        config: dict,
        load_state: bool=False,
        path: str=None,
        model_path: str=None,
    ):
    config = copy.deepcopy(config)
    return _init_from_config(
        config,
        load_state,
        path,
        model_path
    )

def _init_from_config(
        config: dict,
        load_state: bool=False,
        path: str=None,
        model_path: str=None,
    ):
    if 'num' in config:
        objs = []
        num = config['num']
        del config['num']
        for _ in range(num):
            # config is modified in the loop, so we need to deep copy it
            copied_config = copy.deepcopy(config)
            objs.append(_init_from_config(copied_config))
        return objs
    else:
        obj_type = config['type']
        obj_params = config['params']
        for key, value in obj_params.items():
            if isinstance(value, dict):
                obj_params[key] = _init_from_config(value)
            if isinstance(value, list):
                obj_list = []
                for v in value:
                    obj_item = _init_from_config(v)
                    if isinstance(obj_item, list):
                        obj_list.extend(obj_item)
                    else:
                        obj_list.append(obj_item)
                obj_params[key] = obj_list
        obj = TransformerObjectRegistry.get(obj_type)(**obj_params)

    if load_state:
        obj = _load_obj_state(obj, config, path, model_path)
    return obj

def _load_obj_state(obj, config, path=None, model_path=None):
    state_path = config.get('state_path', '')
    if model_path: state_path = model_path
    if path: path = os.path.join(path, state_path)
    obj.load_state(path)
    return obj
