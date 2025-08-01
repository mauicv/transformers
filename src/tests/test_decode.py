import torch
from pytfex.models import (
    get_model,
    GPTBasicConfig,
)
from pytfex.transformer.mask import get_causal_mask


def decode(model, config, input_ids, temp=1, limit=16, sample=True):
    all_preds = []
    for _ in range(limit):
        input_ids = input_ids[:, -config.blk_size:]
        b, l = input_ids.shape
        mask = get_causal_mask(l)
        preds = model(input_ids, mask=mask)
        y = (preds[:, -1, :] / temp).softmax(dim=-1)
        all_preds.append(y)
        if sample:
            next_token = torch.multinomial(y, 1)
        else:
            next_token = torch.argmax(y, dim=-1)

        if not sample: next_token = next_token[None]
        input_ids = torch.cat((input_ids, next_token), dim=-1)
    return input_ids, all_preds


def test_decode(training_setup):
    _, model, _, config = training_setup
    if config.model_type == 'gpt-gumbel-sm-rel-attn':
        for layer in model.layers:
            layer.attn.set_tau(0.5)
            layer.attn.set_hard(True)

    input_ids = torch.tensor([[0, 1]])
    result, _ = decode(model, config, input_ids, limit=20)


def test_kv_cache():
    config = GPTBasicConfig(
        vcb_size=3,
        hdn_dim=3,
        num_heads=1,
        blk_size=3,
        batch_size=1,
        dropout=0.0,
    )
    model = get_model(config)
    model.eval()

    input_ids = torch.tensor([[1]])
    mask = get_causal_mask(1)
    _, kv_cache_1 = model(input_ids, mask=mask, use_kv_cache=True, kv_cache=None)
    input_ids = torch.tensor([[1, 0]])
    mask = get_causal_mask(2)
    preds_1, kv_cache_2 = model(input_ids, mask=mask, use_kv_cache=True, kv_cache=None)
    input_ids = torch.tensor([[0]])
    preds_2, kv_cache_1 = model(input_ids, use_kv_cache=True, kv_cache=kv_cache_1)
    for cache_1, cache_2 in zip(kv_cache_1, kv_cache_2):
        assert torch.allclose(cache_1['v'], cache_2['v'])
        assert torch.allclose(cache_1['k'], cache_2['k'])
    assert torch.allclose(preds_1[:, -1, :], preds_2[:, -1, :])

def test_kv_cache_decode(training_setup):
    _, model, _, config = training_setup
    model.eval()
    if config.model_type == 'gpt-gumbel-sm-rel-attn':
        for layer in model.layers:
            layer.attn.set_tau(0.0)
            layer.attn.set_hard(True)

    input_ids = torch.tensor([[0]])
    result, all_preds = decode(model, config, input_ids, limit=8, sample=False)

    kv_cache=None
    x = input_ids
    all_preds_2 = []
    for _ in range(8):
        preds, kv_cache = model(x, use_kv_cache=True, kv_cache=kv_cache)
        y = (preds[:, [-1], :] / 1).softmax(dim=-1)
        all_preds_2.append(y)
        x = torch.argmax(y, dim=-1)
        input_ids = torch.cat((input_ids, x), dim=-1)
    assert torch.allclose(result, input_ids), f"{config.model_type} fails"

    for a, b in zip(all_preds, all_preds_2):
        # print(a, b)
        assert torch.allclose(a, b), f"{config.model_type} fails"