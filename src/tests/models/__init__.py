from tests.models.mof import get_mof_gpt_config
from tests.models.moe import get_moe_gpt_config
from tests.models.basic import get_basic_gpt_config
from pytfex.transformer.make_model import init_from_yml_string
from dataclasses import dataclass

@dataclass
class GPTMoFConfig:
    """Class for keeping track of an item in inventory."""
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
    """Class for keeping track of an item in inventory."""
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
    """Class for keeping track of an item in inventory."""
    model_type: str = 'gpt-basic'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    batch_size: int = 32
    num_layers: int = 2


def get_model(config):
    config_str = {
        'gpt-mof': get_mof_gpt_config,
        'gpt-moe': get_moe_gpt_config,
        'gpt-basic': get_basic_gpt_config
    }[config.model_type](config)
    return init_from_yml_string(config_str)
