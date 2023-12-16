import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_tuple(x):
    x = _parse_string_to_tuple(x)
    if not isinstance(x, (list, tuple)):
        return (x, x)
    return x


def _parse_string_to_tuple(x):
    """For parsing config strings of the form: '1,2' to a tuple: (1,2)"""
    if not isinstance(x, str):
        return x
    return tuple(map(int, x.split(',')))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)