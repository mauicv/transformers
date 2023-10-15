from src.model.make_model import init_from_config as init_model_from_config
from src.experiment.make_experiment import init_from_config as init_exp_from_config
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import yaml
import os


def test_model(tmpdir):
    with open('./tests/test_yaml/simple.yml', 'r') as file:
        config = yaml.safe_load(file)
    model = init_model_from_config(config)

    with open('./tests/test_yaml/experiment.yml', 'r') as file:
        config = yaml.safe_load(file)
    experiment_1 = init_exp_from_config(config, model)
    assert isinstance(experiment_1.opt, AdamW)
    weight_decays = [
        group['weight_decay'] for group
        in experiment_1.opt.param_groups
    ]
    assert weight_decays == [0.1, 0.0]
    assert isinstance(experiment_1.loss_fn, CrossEntropyLoss)

    experiment_1.save_state(os.path.join(tmpdir, 'exp-state.pt'))
    
    with open('./tests/test_yaml/experiment.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    experiment_2 = init_exp_from_config(
        config,
        model,
        load_state=True,
        path=os.path.join(tmpdir, 'exp-state.pt')
    )

    for opt_1, opt_2 in zip(
                experiment_1.opt.state_dict()['param_groups'],
                experiment_2.opt.state_dict()['param_groups']
            ):
        for key in opt_1.keys():
            assert opt_1[key] == opt_2[key]