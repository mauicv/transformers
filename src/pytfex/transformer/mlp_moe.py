from typing import List
import torch


class MoEMLP(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            experts: List,
            c: int = 2,
        ):
        """Mixture of Expert - expert choice routing MLP layer

        See https://arxiv.org/pdf/2202.09368.pdf for more details.

        Args:
            hidden_dim (int): hidden dimension
            c (int, optional): Capacity of each expert. The capacity factor c denotes on average how
                many experts are utilized by a token. Defaults to 2.
            experts (List, optional): List of experts. Each expert is a torch.nn.Module.
        """
        super(MoEMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.c = c
        self.experts = torch.nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.gate = torch.nn.Linear(
            hidden_dim,
            self.num_experts,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        b, l, *_ = x.shape
        k = self._compute_k(l)
        S = torch.softmax(self.gate(x), dim=-1)
        S = S.transpose(1, 2) # (batch_size, num_experts, tokens)
        G, I = torch.topk(S, k, dim=-1)
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
            ex = x[batch_indices, indices]
            ex_pred = scores[:, :, None] * expert(ex)
            new_x[batch_indices, indices] += ex_pred
        return x

    def _compute_k(self, l: int) -> int:
        k = int((l * self.c) / self.num_experts)
        k = min(max(k, 1), l)
        return k