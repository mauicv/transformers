from pytfex.utils import set_seed
from tests.models.basic_model import get_basic_gpt_config
from tests.models.moe_model import get_moe_gpt_config
from tests.models.mof_model import get_mof_gpt_config
from pytfex.transformer.make_model import init_from_yml_string

import pytest


@pytest.mark.parametrize('model_type,vcb_size,hdn_dim,blk_size,k,num_experts,num_groups', [
    ('gpt-basic', 32, 12, 11, None, None, None),
    ('gpt-moe', 32, 12, 11, 2, 4, None),
    ('gpt-mof', 32, 12, 11, 2, None, 3),
])
def test_train(model_type, vcb_size, hdn_dim, blk_size, k, num_experts, num_groups):
    set_seed(0)
    config = {
        'gpt-basic': lambda: get_basic_gpt_config(vcb_size, hdn_dim, blk_size),
        'gpt-moe': lambda: get_moe_gpt_config(vcb_size, hdn_dim, blk_size, k, num_experts),
        'gpt-mof': lambda: get_mof_gpt_config(vcb_size, hdn_dim, blk_size, k, num_groups)
    }[model_type]()
    model = init_from_yml_string(config)
    assert model