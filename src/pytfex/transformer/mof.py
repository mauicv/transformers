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
        self.gate = torch.nn.Linear(
            hidden_dim,
            self.g,
            bias=False
        )

    def forward(self, x: torch.Tensor, model_kwargs: Dict = None) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
            model_kwargs (Dict): Keyword arguments for the model
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        if model_kwargs is None: model_kwargs = {}
        b, l, *_ = x.shape
        S = torch.softmax(self.gate(x), dim=-1)
        G, I = torch.topk(S, self.k, dim=-1)
        G = G.squeeze(-1)
        I = I.squeeze(-1)
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
        _x = x[batch_indices, element_indices, I]
        _x = G[:, :, :, None] * _x
        _x = _x.reshape(b, l, self.k*self.hidden_dim_per_group)
        _x = self.model(_x, *model_kwargs)
        _x = _x.reshape(b, l, self.k, self.hidden_dim_per_group)
        x[batch_indices, element_indices, I] = _x
        return x.reshape(b, l, -1)
