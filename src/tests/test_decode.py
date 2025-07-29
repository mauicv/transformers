import torch
from pytfex.transformer.mask import get_causal_mask


def decode(model, config, input_ids, temp=1, limit=16, sample=True):
    if config.model_type == 'gpt-gumbel-sm-rel-attn':
        for layer in model.layers:
            layer.attn.set_tau(0.5)
            layer.attn.set_hard(True)

    for _ in range(limit):
        input_ids = input_ids[:, -config.blk_size:]
        mask = get_causal_mask(input_ids.shape[1])
        preds = model(input_ids, mask=mask)
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
    input_ids = torch.tensor([[0, 1]])
    result = decode(model, config, input_ids, limit=20)
    print(result)