from pytfex.utils import set_seed
from tests.models.mof import get_mof_gpt_config
from pytfex.transformer.make_model import init_from_yml_string
import torch
import cProfile


set_seed(0)
model_type = 'gpt-mof'
vcb_size=65
hdn_dim=256
blk_size=256
k=2
num_proj=4
batch_size=32
num_layers=1
config = get_mof_gpt_config(vcb_size, hdn_dim, blk_size, k, num_proj, num_layers)
model = init_from_yml_string(config)

t1 = torch.randint(0, vcb_size, (batch_size, blk_size))

with cProfile.Profile() as pr:
    output = model(t1)

pr.print_stats()
pr.dump_stats('./src/benchmarks/results/mof.prof')