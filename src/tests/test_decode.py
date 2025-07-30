import torch
from pytfex.transformer.mask import get_causal_mask


def decode(model, config, input_ids, temp=1, limit=16, sample=True):
    for _ in range(limit):
        input_ids = input_ids[:, -config.blk_size:]
        preds = model(input_ids)
        y = (preds[:, -1, :] / temp).softmax(dim=-1)
        if sample:
            next_token = torch.multinomial(y, 1)
        else:
            next_token = torch.argmax(y, dim=-1)

        if not sample: next_token = next_token[None]
        input_ids = torch.cat((input_ids, next_token), dim=-1)
    return input_ids


def test_decode(training_setup):
    _, model, _, config = training_setup
    if config.model_type == 'gpt-gumbel-sm-rel-attn':
        for layer in model.layers:
            layer.attn.set_tau(0.5)
            layer.attn.set_hard(True)

    input_ids = torch.tensor([[0, 1]])
    result = decode(model, config, input_ids, limit=20)
    print(result)


def test_kv_cache(training_setup):
    _, model, _, config = training_setup
    model.eval()
    if config.model_type == 'gpt-gumbel-sm-rel-attn':
        for layer in model.layers:
            layer.attn.set_tau(0.0)
            layer.attn.set_hard(True)


    input_ids = torch.tensor([[0, 1]])
    
    preds_1 = model(input_ids)
    preds_2, kv_cache = model(input_ids, use_kv_cache=True, kv_cache=None)
    assert torch.allclose(preds_1, preds_2)
    for cache in kv_cache:
        assert cache['k'].shape == (1, 4, 2, 64)
        assert cache['v'].shape == (1, 4, 2, 64)


def test_kv_cache_decode(training_setup):
    _, model, _, config = training_setup
    model.eval()
    if config.model_type == 'gpt-gumbel-sm-rel-attn':
        for layer in model.layers:
            layer.attn.set_tau(0.0)
            layer.attn.set_hard(True)

    input_ids = torch.tensor([[0, 1]])
    result = decode(model, config, input_ids, limit=8, sample=False)

    kv_cache=[{}, {}]
    x = input_ids
    for _ in range(8):
        preds, kv_cache = model(x, use_kv_cache=True, kv_cache=kv_cache)
        y = (preds[:, [-1], :] / 1).softmax(dim=-1)
        x = torch.argmax(y, dim=-1)
        input_ids = torch.cat((input_ids, x), dim=-1)
    print(result)
    print(input_ids)
    assert torch.allclose(result, input_ids)