from torch.utils.data.dataloader import DataLoader
from tests.dataset import SortDataset

from pytfex.transformer.make_model import init_from_yml_string
from pytfex.utils import set_seed
# from tests.models.basic import get_basic_gpt_config
# from tests.models.moe import get_moe_gpt_config
# from tests.models.mof import get_mof_gpt_config
from tests.models import get_model, GPTMoFConfig, GPTMoEConfig, GPTBasicConfig

import torch
import pytest


@pytest.fixture(params=[
    # (model_type, hdn_dim, length, num_digits, batch_size, _, _, _)
    (GPTBasicConfig(
        model_type='gpt-basic',
        vcb_size=3,
        hdn_dim=256,
        blk_size=12,
        batch_size=32,
    ), 6),
    # (model_type, hdn_dim, length, num_digits, batch_size, k, num_experts, _)
    (GPTMoEConfig(
        model_type='gpt-moe',
        vcb_size=3,
        hdn_dim=256,
        blk_size=12,
        c=2,
        num_experts=4,
        batch_size=32,
    ), 6),
    # (model_type, hdn_dim, length, num_digits, batch_size, k, _, num_groups)
    (GPTMoFConfig(
        model_type='gpt-mof',
        vcb_size=3,
        hdn_dim=256,
        blk_size=12,
        k=2,
        num_proj=4,
        batch_size=32,
    ), 6)
])
def training_setup(request):
    set_seed(0)
    config, length = request.param
    num_digits = config.vcb_size
    ds = SortDataset(split='train', length=length, num_digits=num_digits)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    model = get_model(config)

    def val_fn(model):
        ds = SortDataset(split='test', length=length, num_digits=num_digits)
        dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        total = 0
        sum_acc = 0
        for x, y_true in dl:
            train_inds = y_true > 0
            y_hat = model(x)
            y_true = y_true[train_inds]
            y_hat = y_hat[train_inds]
            r = torch.eq(y_true, y_hat.argmax(dim=-1))
            l = r.shape[0]
            total += l
            sum_acc += r.sum()
        acc = sum_acc / total
        return acc

    return dl, model, val_fn, config.model_type