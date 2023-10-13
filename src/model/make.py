from src.model.layer import TransformerLayer
from src.model.attention import Attention
from src.model.mlp import MLP
from src.model.base import Transformer
import yaml


def init_from_config(config):
    return _init_obj(config)


def _init_obj(config):
    if 'num' in config:
        objs = []
        num = config['num']
        del config['num']
        for _ in range(num):
            objs.append(_init_obj(config))
        return objs
    else:
        obj_type = config['type']
        obj_params = config['params']
        for key, value in obj_params.items():
            if isinstance(value, dict):
                obj_params[key] = _init_obj(value)
            if isinstance(value, list):
                obj_list = []
                for v in value:
                    obj_item = _init_obj(v)
                    if isinstance(obj_item, list):
                        obj_list.extend(obj_item)
                    else:
                        obj_list.append(obj_item)
                obj_params[key] = obj_list
        return globals()[obj_type](**obj_params)


def init_from_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return _init_obj(config)