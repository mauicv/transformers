from tests.models.mof import get_mof_gpt_config
from tests.models.moe import get_moe_gpt_config
from tests.models.basic import get_basic_gpt_config
from tests.models.moemof import get_moemof_gpt_config
from pytfex.transformer.make_model import init_from_yml_string
from dataclasses import dataclass

@dataclass
class GPTMoFConfig:
    model_type: str = 'gpt-mof'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    k: int = 2
    num_proj: int = 4
    batch_size: int = 32
    num_layers: int = 2


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


@dataclass
class GPTBasicConfig:
    model_type: str = 'gpt-basic'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    batch_size: int = 32
    num_layers: int = 2


@dataclass
class GPTMoEMoFConfig:
    model_type: str = 'gpt-moemof'
    vcb_size: int = 65
    hdn_dim: int = 4*256
    blk_size: int = 256
    c: int = 2
    num_experts: int = 4
    k: int = 2
    num_proj: int = 4
    batch_size: int = 32
    num_layers: int = 2


def get_model(config):
    config_str = {
        'gpt-mof': get_mof_gpt_config,
        'gpt-moe': get_moe_gpt_config,
        'gpt-basic': get_basic_gpt_config,
        'gpt-moemof': get_moemof_gpt_config
    }[config.model_type](config)
    return init_from_yml_string(config_str)
