import torch


def gumbel_softmax(logits, tau=1.0, eps=1e-20, hard=False):
    if hard:
        y_soft = torch.softmax(logits, dim=-1)
        i = y_soft.argmax(dim=-1)
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(
            dim=-1,
            index=i.unsqueeze(-1),
            src=torch.ones_like(y_soft),
        )
        y_masked = y_soft * y_hard
        y_hard = (y_hard - y_masked).detach() + y_masked
        return y_hard
    else:
        s = torch.rand_like(logits)
        gumbels = -torch.log(-torch.log(s + eps) + eps)
        gumbels = (logits + gumbels) / tau
        y_soft = torch.softmax(gumbels, dim=-1)
        return y_soft