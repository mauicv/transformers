from typing import List
import torch


class MoEMLP(torch.nn.Module):
    """Mixture of Expert - expert choice routing MLP layer

    See https://arxiv.org/pdf/2202.09368.pdf for details
    """
    def __init__(
            self,
            hidden_dim: int,
            k: int,
            experts: List=None
        ):
        super(MoEMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.k = k
        self.experts = torch.nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.gate = torch.nn.Linear(
            hidden_dim,
            self.num_experts,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, *_ = x.shape
        S = torch.softmax(self.gate(x), dim=-1)
        S = S.transpose(1, 2) # (batch_size, num_experts, tokens)
        G, I = torch.topk(S, self.k, dim=-1)
        # I - (batch_size, num_experts, top_k_tokens) - indices
        # G - (batch_size, num_experts, top_k_tokens) - weights
        new_x = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            indices = I[:, i]
            scores = G[:, i]
            batch_indices = (torch
                .arange(b)
                .view(-1, 1)
                .expand_as(indices)
            )
            # (batch_size, top_k_tokens, hidden_dim) - tokens for expert i
            # NOTE: not sure this is efficient? maybe use gather?
            ex = x[batch_indices, indices]
            ex_pred = scores[:, :, None] * expert(ex)
            new_x[batch_indices, indices] += ex_pred
        return x
