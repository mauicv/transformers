"""
Some work here is derived from: https://github.com/huggingface/transformers/blob/c7f076a00ee54f777b3d3322c91bc11489a47950/src/transformers/models/mixtral/modeling_mixtral.py#L655
"""

from typing import Dict
import torch


class MoF(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            model: torch.nn.Module,
            num_groups: int = 2,
            k: int = 2,
        ):
        """Mixture of Features (MoF) layer

        Selects k subgroups groups of features from the input tensor and applies a model to them.

        Args:
            hidden_dim (int): hidden dimension
            model (torch.nn.Module): Model that is applied to the selected dimensions
            num_groups (int, optional): Number of feature groups. Defaults to 2.
            k (int, optional): Number of groups to select. Defaults to 1.
        """
        super(MoF, self).__init__()
        assert hidden_dim % num_groups == 0, "num_groups must divide hidden_dim"
        assert k <= num_groups, "k must be less than or equal to num_groups"
        self.hidden_dim = hidden_dim
        self.g = num_groups
        self.k = k
        self.hidden_dim_per_group = (hidden_dim // num_groups)
        self.model = model
        self.source_gate = torch.nn.Linear(
            hidden_dim,
            self.g,
            bias=False
        )
        self.dest_gate = torch.nn.Linear(
            hidden_dim,
            self.g,
            bias=False
        )

    def _compute_source_scores(self, x: torch.Tensor) -> torch.Tensor:
        S = torch.sigmoid(self.source_gate(x))
        G, I = torch.topk(S, self.k, dim=-1)
        return G.squeeze(-1), I.squeeze(-1)

    def _compute_dest_scores(self, x: torch.Tensor) -> torch.Tensor:
        S = torch.sigmoid(self.dest_gate(x))
        G, I = torch.topk(S, self.k, dim=-1)
        return G.squeeze(-1), I.squeeze(-1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
            model_kwargs (Dict): Keyword arguments for the model

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        b, l, *_ = x.shape
        G_source, I_source = self._compute_source_scores(x)
        G_dest, I_dest = self._compute_dest_scores(x)
        x = x.reshape(b, l, self.g, -1)
        batch_indices = (torch
            .arange(b)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, l, self.k)
        )
        element_indices = (torch
            .arange(l)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(b, -1, self.k)
        )
        z = torch.zeros_like(x)
        _x = x[batch_indices, element_indices, I_source]
        _x = G_source[:, :, :, None] * _x
        _x = _x.reshape(b, l, self.k*self.hidden_dim_per_group)
        _x = self.model(_x, **kwargs)
        _x = _x.reshape(b, l, self.k, self.hidden_dim_per_group)
        z[batch_indices, element_indices, I_dest] = G_dest[:, :, :, None] * _x
        return z.reshape(b, l, -1)


class MoF2(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            model: torch.nn.Module,
            num_proj: int = 2,
            k: int = 2,
        ):
        """Mixture of Features-2 (MoF2) layer

        mof-2-runtime: O(d * k + 2*(d*d)/k + 8*d/k*d/k)
        """
        super(MoF2, self).__init__()
        assert hidden_dim % num_proj == 0, "num_groups must divide hidden_dim"
        assert k <= num_proj, "k must be less than or equal to num_groups"
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
        return G.squeeze(-1), I.squeeze(-1)

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

        proj_mask = torch.nn.functional.one_hot(
            I, num_classes=self.num_proj
        ).permute(2, 1, 0)

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