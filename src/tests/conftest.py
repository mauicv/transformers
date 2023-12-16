from torch.utils.data.dataloader import DataLoader
from tests.dataset import SortDataset

from pytfex.transformer.make_model import init_from_yml_string
from pytfex.utils import set_seed
from tests.basic_model import get_basic_gpt_config
from tests.moe_model import get_moe_gpt_config

import torch

import pytest


@pytest.fixture(params=[
    (256, 6, 3, 32, None, None, 'gpt-basic'), # (hdn_dim, length, num_digits, batch_size, _, _, model_type)
    (256, 6, 3, 32, 2, 4, 'gpt-moe') # (hdn_dim, length, num_digits, batch_size, k, num_experts, model_type)
])
def training_setup(request):
    set_seed(0)

    hdn_dim, length, num_digits, batch_size, k, num_experts, model_type = request.param
    ds = SortDataset(split='train', length=length, num_digits=num_digits)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    blk_size = ds.get_block_size()
    vcb_size = ds.get_vocab_size()

    config = {
        'gpt-basic': lambda: get_basic_gpt_config(vcb_size, hdn_dim, blk_size),
        'gpt-moe': lambda: get_moe_gpt_config(vcb_size, hdn_dim, blk_size, k, num_experts)
    }[model_type]()
    model = init_from_yml_string(config)

    def val_fn(model):
        ds = SortDataset(split='test', length=length, num_digits=num_digits)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
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

    return dl, model, val_fn, model_type