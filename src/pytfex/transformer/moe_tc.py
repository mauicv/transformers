from typing import List
import torch


class TokenChoiceMoE(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            experts: List,
            k: int = 2,
        ):
        """Mixture of Expert - token choice routing layer

        See https://arxiv.org/pdf/1701.06538.pdf for more details.

        Implementation based on 
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L773

        Args:
            hidden_dim (int): hidden dimension
            k (int, optional): Number of experts to route to. Defaults to 2.
            experts (List, optional): List of experts. Each expert is a torch.nn.Module.
        """
        super(TokenChoiceMoE, self).__init__()
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
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        b, l, hdn_dim = x.shape
        x = x.view(b * l, hdn_dim)
        S = torch.sigmoid(self.gate(x))
        G, I = torch.topk(S, self.k, dim=-1)
        final_x = torch.zeros(
            (b * l, hdn_dim),
            dtype=x.dtype,
            device=x.device
        )

        expert_mask = torch.nn.functional.one_hot(
            I, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for i, expert in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[i])
            if top_x.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = x[None, top_x_list] \
                .reshape(-1, hdn_dim)
            current_hidden_states = expert(current_state) \
                * G[top_x_list, idx_list, None]
            final_x.index_add_(
                0, top_x,
                current_hidden_states.to(x.dtype)
            )
        final_x = final_x.reshape(b, l, hdn_dim)
        return final_x
