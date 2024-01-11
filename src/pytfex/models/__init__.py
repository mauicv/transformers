from pytfex.models.moe import get_moe_gpt_config
from pytfex.models.basic import get_basic_gpt_config
from pytfex.models.sparse_mlp import get_sparse_mlp_gpt_config
from pytfex.transformer.make_model import init_from_yml_string
from dataclasses import dataclass


@dataclass
class GPTMoEConfig:
    model_type: str = 'gpt-moe'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    c: int = 2
    num_experts: int = 4
    batch_size: int = 32
    num_layers: int = 2
    mlp_hdn_dim: int = 1024
    dropout: float = 0.1
    num_heads: int = 4


@dataclass
class GPTBasicConfig:
    model_type: str = 'gpt-basic'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    batch_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    num_heads: int = 4


@dataclass
class SparseMLPGPTConfig:
    model_type: str = 'sparse-mlp-gpt'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    batch_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    num_heads: int = 4
    k: int = 2


def get_model(config):
    config_str = {
        'gpt-moe': get_moe_gpt_config,
        'gpt-basic': get_basic_gpt_config,
        'sparse-mlp-gpt': get_sparse_mlp_gpt_config,
    }[config.model_type](config)
    return init_from_yml_string(config_str)
