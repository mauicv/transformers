from pytfex.utils import set_seed, count_parameters

import torch
import time
import pytest


@pytest.mark.skip(reason="Slow running/intermittent test")
def test_train(training_setup):
    set_seed(0)
    dl, model, val_fn, model_type = training_setup
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    acc = val_fn(model)

    print('\n')
    print(f'-- model-type : {model_type} --')
    print(f'-- # params   : {count_parameters(model)} --')
    print('epoch_|_loss_____|_acc______|_time____')
    print(f'    -1| None     | {acc:0.5}  | None')
    for epoch in range(10):
        s = time.time()
        for x, y_true in dl:
            b, l = x.shape
            opt.zero_grad()
            y = model(x)
            y, y_true = y.reshape(b*l, -1), y_true.reshape(b*l)
            train_inds = y_true > 0
            y_true = y_true[train_inds]
            y = y[train_inds]
            loss = loss_fn(y, y_true)
            loss.backward()
            opt.step()
        e = time.time()
        acc = val_fn(model)
        print(f'{epoch:>6}| {loss.item():<8.5} | {acc:0.5}  | {e-s:0.5}')

    assert loss.item() < 0.15
    acc = val_fn(model)
    assert acc > 0.95
