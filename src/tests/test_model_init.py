from pytfex.utils import set_seed
from tests.basic_model import get_basic_gpt_config
from tests.moe_model import get_moe_gpt_config
from pytfex.transformer.make_model import init_from_yml_string

import pytest


@pytest.mark.parametrize('vcb_size,hdn_dim,blk_size,k,num_experts,model_type', [
    (32, 12, 11, None, None, 'gpt-basic'),
    (32, 12, 11, 2, 4, 'gpt-moe')
])
def test_train(vcb_size, hdn_dim, blk_size, k, num_experts, model_type):
    set_seed(0)
    config = {
        'gpt-basic': get_basic_gpt_config(vcb_size, hdn_dim, blk_size),
        'gpt-moe': get_moe_gpt_config(vcb_size, hdn_dim, blk_size, k, num_experts)
    }[model_type]
    model = init_from_yml_string(config)
    print(model)
    assert model