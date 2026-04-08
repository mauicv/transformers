from typing import List
import torch


def generate_relative_positions(L):
    positions = torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
    return positions


class ExpertChoiceSoup(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            experts: List,
            c: int = 2,
            output_dim: int = None,
        ):
        """Mixture of Expert - expert choice routing layer

        See https://arxiv.org/pdf/2202.09368.pdf for more details.

        Args:
            hidden_dim (int): hidden dimension
            c (int, optional): Capacity of each expert. The capacity factor c denotes on average how
                many experts are utilized by a token. Defaults to 2.
            experts (List, optional): List of experts. Each expert is a torch.nn.Module.
        """
        super(ExpertChoiceSoup, self).__init__()
        self.hidden_dim = hidden_dim
        self.c = c
        self.experts = torch.nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.gate = torch.nn.Linear(
            hidden_dim,
            self.num_experts,
            bias=False
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        b, l, hdn_dim = x.shape
        x = x.view(b * l, hdn_dim)
        k = self._compute_k(l*b)
        S = torch.sigmoid(self.gate(x))
        S = S.transpose(0, 1) # (num_experts, batch_size*tokens)
        G, I = torch.topk(S, k, dim=-1)
        # I - (num_experts, top_k_tokens) - indices
        # G - (num_experts, top_k_tokens) - weights
        if self.output_dim is not None:
            new_x = torch.zeros(b * l, self.output_dim, device=x.device)
        else:
            new_x = torch.zeros_like(x, device=x.device)
        for i, expert in enumerate(self.experts):
            indices = I[i]
            scores = G[i]
            ex = x[indices]
            ex_pred = scores[:, None] * expert(ex)
            new_x[indices] += ex_pred
        if self.output_dim is not None:
            new_x = new_x.view(b, l, self.output_dim)
        else:
            new_x = new_x.view(b, l, hdn_dim)
        return new_x

    def _compute_k(self, l: int) -> int:
        k = int((l * self.c) / self.num_experts)
        k = min(max(k, 1), l)
        return k


class RelativeAttentionSoup(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_experts: int,
            num_positions: int,
            dropout=0.1
        ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = torch.tensor(
            dropout,
            dtype=torch.float32
        )
        self.attn_dropout = torch.nn.Dropout(self.dropout)
        self.resid_dropout = torch.nn.Dropout(self.dropout)

        self.qkv = ExpertChoiceSoup(
            experts=[
                torch.nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
                for _ in range(num_experts)
            ],
            hidden_dim=self.hidden_dim,
            output_dim=3*self.hidden_dim,
            c=2,
        )
        self.linear = ExpertChoiceSoup(
            hidden_dim=self.hidden_dim,
            experts=[
                torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(num_experts)
            ],
            c=2,
        )
        self.head_dim = self.hidden_dim // self.num_heads
        self.attn_drop = torch.nn.Dropout(0.1)
        self.resid_drop = torch.nn.Dropout(0.1)

        self.num_positions = num_positions
        self.Er = torch.nn.Parameter(torch.randn(
            self.num_heads, 
            self.num_positions,
            self.head_dim
        ))

    def forward(
            self,
            x,
            mask=None,
        ):
        b, l, d = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=2)
        q = q.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        b, _, kv_l, _ = v.shape
        rel_pos = generate_relative_positions(kv_l)
        Er = self.Er[:, rel_pos].unsqueeze(0)
        # compute attention
        hd = torch.tensor(self.head_dim, dtype=torch.float32)
        a = q @ k.transpose(-2, -1) / torch.sqrt(hd)
        QEr = torch.einsum('bnlh,rnlkh->bnlk', q, Er)
        a = a + QEr

        if mask is not None:
            a = a.masked_fill(mask, float('-inf'))
        a = torch.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        output =  (a @ v).transpose(1, 2).reshape(b, l, d)
        output = self.linear(output)
        output = self.resid_dropout(output)
        return output