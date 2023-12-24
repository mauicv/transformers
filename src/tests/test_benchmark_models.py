from pytfex.utils import set_seed
from tests.models.basic import get_basic_gpt_config
from tests.models.moe import get_moe_gpt_config
from tests.models.mof import get_mof_gpt_config
from tests.models.mof2 import get_mof2_gpt_config
from pytfex.transformer.make_model import init_from_yml_string

import pytest
import torch.nn as nn
import torch
import time


@pytest.mark.parametrize('model_type,vcb_size,hdn_dim,blk_size,k,num_experts,num_groups', [
    ('gpt-basic', 65, 256, 256, None, None, None),
    ('gpt-moe', 65, 256, 256, 2, 4, None),
    ('gpt-mof2', 65, 256, 256, 2, 2, 4),
])
@pytest.mark.parametrize('with_train', [False])
@pytest.mark.parametrize('iterations', [250])
@pytest.mark.parametrize('batch_size', [6])
@pytest.mark.parametrize('num_layers', [1])
@pytest.mark.parametrize('with_cuda', [False])
def test_train(
        model_type,
        vcb_size,
        hdn_dim,
        blk_size,
        k,
        num_experts,
        num_groups,
        with_train,
        iterations,
        batch_size,
        num_layers,
        with_cuda
    ):
    set_seed(0)
    print('\n\n')
    config = {
        'gpt-basic': lambda: get_basic_gpt_config(vcb_size, hdn_dim, blk_size, num_layers),
        'gpt-moe': lambda: get_moe_gpt_config(vcb_size, hdn_dim, blk_size, k, num_experts, num_layers),
        'gpt-mof': lambda: get_mof_gpt_config(vcb_size, hdn_dim, blk_size, k, num_groups, num_layers),
        'gpt-mof2': lambda: get_mof2_gpt_config(vcb_size, hdn_dim, blk_size, k, num_groups, num_layers)
    }[model_type]()
    model = init_from_yml_string(config)
    loss_function = nn.CrossEntropyLoss()
    assert model
    if with_cuda: model = model.cuda()
    times = []
    for _ in range(iterations):
        t1 = torch.randint(0, vcb_size, (batch_size, blk_size))
        if with_cuda:
            t1 = t1.cuda()
            torch.cuda.synchronize()
        start_epoch = time.time()
        output = model(t1)
        if with_train:
            t2 = torch.randint(0, vcb_size, (batch_size, blk_size))
            if with_cuda: t2 = t2.cuda()
            b, l, _ = output.shape
            l = loss_function(output.reshape(b*l, -1), t2.reshape(b*l))
            l.backward()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)

    print('-------------------')
    print(f'model_type: {model_type} - with_train: {with_train}')
    print(f'avg: {sum(times)/len(times)}')
    print(f'num_param: {sum(p.numel() for p in model.parameters())}')