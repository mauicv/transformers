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

        Selects k subgroups groups of features from the input tensor and applies a model to them.

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
        self.projs = torch.nn.ModuleList([
            torch.nn.Linear(
                self.hidden_dim,
                self.hidden_dim_per_group,
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
            model_kwargs (Dict): Keyword arguments for the model

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        b, l, *_ = x.shape
        G, I = self._compute_scores(x)
        print(x.shape, I.shape, I)
        for ind, proj in enumerate(self.projs):
            pass


        # batch_indices = (torch
        #     .arange(b)
        #     .unsqueeze(1)
        #     .unsqueeze(2)
        #     .expand(-1, l, self.k)
        # )
        # element_indices = (torch
        #     .arange(l)
        #     .unsqueeze(0)
        #     .unsqueeze(2)
        #     .expand(b, -1, self.k)
        # )
        # z = torch.zeros_like(x)
        # _x = x[batch_indices, element_indices, I_source]
        # _x = G_source[:, :, :, None] * _x
        # _x = _x.reshape(b, l, self.k*self.hidden_dim_per_group)
        # _x = self.model(_x, **kwargs)
        # _x = _x.reshape(b, l, self.k, self.hidden_dim_per_group)
        # z[batch_indices, element_indices, I_dest] = G_dest[:, :, :, None] * _x
        # return z.reshape(b, l, -1)
