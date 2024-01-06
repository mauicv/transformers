"""
Some work here is derived from: https://github.com/huggingface/transformers/blob/c7f076a00ee54f777b3d3322c91bc11489a47950/src/transformers/models/mixtral/modeling_mixtral.py#L655
"""

from typing import Dict
import torch
import time


class MoF(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            model: torch.nn.Module,
            num_proj: int = 2,
            k: int = 2,
        ):
        """Mixture of Features (MoF) layer

        runtime: O(d * k + 2*(d*d)/k + 8*d/k*d/k)
        """
        super(MoF, self).__init__()
        assert hidden_dim % num_proj == 0, "num_proj must divide hidden_dim"
        assert k <= num_proj, "k must be less than or equal to num_proj"
        self.hidden_dim = hidden_dim
        self.num_proj = num_proj
        self.k = k
        self.model = model
        self.gate = torch.nn.Linear(
            hidden_dim,
            self.num_proj,
            bias=False
        )
        self.hidden_dim_per_group = (hidden_dim // num_proj)
        self.down_projs = torch.nn.ModuleList([
            torch.nn.Linear(
                self.hidden_dim,
                self.hidden_dim_per_group,
                bias=False
            ) for _ in range(self.num_proj)
        ])
        self.up_projs = torch.nn.ModuleList([
            torch.nn.Linear(
                self.hidden_dim_per_group,
                self.hidden_dim,
                bias=False
            ) for _ in range(self.num_proj)
        ])


    def _compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        S = torch.sigmoid(self.gate(x))
        G, I = torch.topk(S, self.k, dim=-1)
        return G, I

    def _project(self, x: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
            kwargs (Dict): Keyword arguments for the model

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        b, l, *_ = x.shape
        x = x.view(-1, self.hidden_dim)

        G, I = self._compute_scores(x)
        down_x = torch.zeros(
            (b * l, self.hidden_dim_per_group),
            device=x.device
        )

        up_x = torch.zeros(
            (b * l, self.hidden_dim),
            device=x.device
        )

        proj_mask = (torch.nn.functional
            .one_hot(I, num_classes=self.num_proj)
            .permute(2, 1, 0)
        )

        for ind, proj in enumerate(self.down_projs):
            idx, top_x = torch.where(proj_mask[ind])
            if top_x.shape[0] == 0:
                continue
            
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = x[None, top_x_list].reshape(-1, self.hidden_dim)
            current_hidden_states = proj(current_state) * G[top_x_list, idx_list, None]

            down_x.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        down_x = down_x.reshape(b, l, self.hidden_dim_per_group)
        down_x = self.model(down_x, **kwargs)
        down_x = down_x.view(-1, self.hidden_dim_per_group)

        for ind, proj in enumerate(self.up_projs):
            idx, top_x = torch.where(proj_mask[ind])
            if top_x.shape[0] == 0:
                continue
            
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = down_x[None, top_x_list].reshape(-1, self.hidden_dim_per_group)
            current_hidden_states = proj(current_state) * G[top_x_list, idx_list, None]

            up_x.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        up_x = up_x.reshape(b, l, self.hidden_dim)
        return up_x