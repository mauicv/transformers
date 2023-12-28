from pytfex.utils import set_seed
from tests.models import get_model, GPTMoFConfig, GPTMoEConfig, GPTBasicConfig
import torch
from profiling import Profiling


benchmarks = [
    GPTBasicConfig(num_layers=1, hdn_dim=2048),
    GPTMoFConfig(num_layers=1, num_proj=8, k=2, hdn_dim=2048),
    # GPTMoEConfig(num_layers=1, num_experts=3, c=1),
]

for config in benchmarks:
    print(config)
    set_seed(0)
    model = get_model(config)

    t1 = torch.randint(0, config.vcb_size, (config.batch_size, config.blk_size))

    model.eval()
    output_1 = model(t1)
    # with Profiling(model.layers[0].mlp) as p:
    #     output_2 = model(t1)

    with Profiling(model) as p:
        output_2 = model(t1)

    assert torch.allclose(output_1, output_2), 'different outputs'

    print(p)