from torch.utils.data.dataloader import DataLoader
from tests.dataset import SortDataset
from src.model.make_model import init_from_yml_string
import torch
import pytest
from src.utils import set_seed
set_seed(0)

def validate(model):
    ds = SortDataset(split='test', length=6, num_digits=3)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
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


@pytest.mark.skip(reason="Slow running/intermittent test")
def test_train():
    ds = SortDataset(split='train', length=6, num_digits=3)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    blk_size = ds.get_block_size()
    vcb_size = ds.get_vocab_size()
    hdn_dim = 256
    config = f"""
    type: 'GPT'
    params:
      dropout: 0.5
      hidden_dim: {hdn_dim}
      max_sequence_length: {blk_size}
      dictionary_size: {vcb_size}
      num_heads: 4
      dropout: 0.5
      head:
        type: 'ClassificationHead'
        params:
          hidden_dim: {hdn_dim}
          vocab_size: {vcb_size}
      layers:
        - num: 2
          type: 'TransformerLayer'
          params:
            hidden_dim: {hdn_dim}
            attn:
              type: 'Attention'
              params:
                hidden_dim: {hdn_dim}
                num_heads: 4
                dropout: 0.5
            mlp:
              type: 'MLP'
              params:
                hidden_dim: {hdn_dim}
                dropout: 0.5
    """

    model = init_from_yml_string(config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    loss_fn = torch.nn.CrossEntropyLoss()

    acc = validate(model)

    print('\n')
    print('epoch_|_loss_____|_acc______')
    print(f'    -1| None     | {acc:0.5}')
    for epoch in range(5):
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

        acc = validate(model)

        print(f'{epoch:>6}| {loss.item():<8.5} | {acc:0.5}')

    assert loss.item() < 0.1
    acc = validate(model)
    assert acc > 0.95
