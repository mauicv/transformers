import yaml

#------------------------------------------------------------------------
# These imports are necessary to make the code work, see globals() below!
from src.experiment.transformer_experiment import TransformerExperiment
from torch.optim import AdamW, Adam
from torch.nn import CrossEntropyLoss
import torch
#------------------------------------------------------------------------

def _parse_kwargs(args):
    types = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
    }

    kwargs = {}
    for arg in args:
        name_type, value = arg.split('=')
        name, type = name_type.split(':')
        kwargs[name] = types[type](value)
    return kwargs

def init_from_config(
        config: dict,
        model: dict[str, torch.nn.Module],
        load_state: bool=False,
        path: str=None,
    ):
    obj_type = config['type']
    obj_params = config['params']
    for key, value in obj_params.items():
        if isinstance(value, dict):
            obj_params[key] = init_from_config(value, model)
        if isinstance(value, str) and 'fn!' == value[:3]:
            fn_name, *args = value[3:].split('|')
            kwargs = _parse_kwargs(args)
            obj_params[key] = getattr(model, fn_name)(**kwargs)
    experiment = globals()[obj_type](**obj_params)
    if load_state: experiment.load_state(path)
    return experiment


def init_from_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return init_from_config(config)

def init_from_yml_string(config_string):
    config = yaml.safe_load(config_string)
    return init_from_config(config)