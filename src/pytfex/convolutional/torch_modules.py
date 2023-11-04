import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch


def get_conv(*args, with_weight_norm=True, **kwargs):
    conv_layer = nn.Conv2d(*args, **kwargs)
    if with_weight_norm:
        conv_layer = weight_norm(conv_layer)
    return conv_layer


def get_norm():
    return nn.BatchNorm2d


def get_rep_pad():
    return nn.ReplicationPad2d

def get_upsample():
    return nn.UpsamplingNearest2d


def swish(x):
    return x*torch.sigmoid(x)


def get_nonlinearity(type='ELU'):
    return {
        'swish': lambda: swish,
        'relu': lambda: torch.nn.ReLU(),
        'leakyrelu': lambda: torch.nn.LeakyReLU(0.2),
        'tanh': lambda: torch.nn.Tanh(),
        'sigmoid': lambda: torch.nn.Sigmoid(),
        'ELU': lambda: torch.nn.ELU(),
    }[type]()
    
    