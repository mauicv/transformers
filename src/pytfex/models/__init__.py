from pytfex.models.ec_moe import get_ec_moe_gpt_config
from pytfex.models.tc_moe import get_tc_moe_gpt_config
from pytfex.models.rel_attn import get_rel_attn_gpt_config
from pytfex.models.gumbel_sm_rel_attn import get_gumbel_sm_rel_attn_gpt_config
from pytfex.models.basic import get_basic_gpt_config
from pytfex.transformer.make_model import init_from_yml_string
from dataclasses import dataclass


@dataclass
class GPTExpertChoiceMoEConfig:
    model_type: str = 'gpt-ec-moe'
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
class GPTTokenChoiceMoEConfig:
    model_type: str = 'gpt-tc-moe'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    k: int = 2
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
class GPTRelAttnConfig:
    model_type: str = 'gpt-rel-attn'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    batch_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    num_heads: int = 4


@dataclass
class GPTGumbelSoftmaxRelativeAttentionConfig:
    model_type: str = 'gpt-gumbel-sm-rel-attn'
    vcb_size: int = 65
    hdn_dim: int = 256
    blk_size: int = 256
    batch_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    num_heads: int = 4


def get_model(config):
    config_str = {
        'gpt-ec-moe': get_ec_moe_gpt_config,
        'gpt-tc-moe': get_tc_moe_gpt_config,
        'gpt-basic': get_basic_gpt_config,
        'gpt-rel-attn': get_rel_attn_gpt_config,
        'gpt-gumbel-sm-rel-attn': get_gumbel_sm_rel_attn_gpt_config
    }[config.model_type](config)
    return init_from_yml_string(config_str)
