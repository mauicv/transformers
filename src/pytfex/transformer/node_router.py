from typing import List, Tuple, Union, Optional
import torch

class RouteTensor:
    def __init__(self,
            data: Union[Tuple[torch.Tensor], List[torch.Tensor], torch.Tensor],
            batch_inds: Optional[torch.Tensor]=None
        ) -> None:

        if isinstance(data, torch.Tensor):
            self.data = data
            self.batch_inds = batch_inds

        if isinstance(data, (list, tuple)):
            inds = [0]
            for d in data:
                inds.append(inds[-1] + len(d))
            self.batch_inds = torch.tensor(inds)
            self.data = torch.cat(data, dim=0)

    def to_batches(self) -> torch.Tensor:
        for i in range(len(self.batch_inds) - 1):
            yield self.data[self.batch_inds[i]:self.batch_inds[i+1]]

    def __getitem__(self, ind):
         return self.data[self.batch_inds[ind]:self.batch_inds[ind+1]]

    def apply(self, func) -> 'RouteTensor':
        return RouteTensor(
            data=func(self.data),
            batch_inds=self.batch_inds
        )

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def shapes(self) -> torch.Size:
        return tuple(t.shape for t in self.to_batches())


class NodeRouter(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_nodes: int,
            num_heads: int,
            k: int,
            dropout: float=0.5,
        ):
        super(NodeRouter, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.k = k

        self.gate_dropout = torch.nn.Dropout(
            torch.tensor(
                dropout,
                dtype=torch.float32
            )
        )

        self.node_dropout = torch.nn.Dropout(
            torch.tensor(
                dropout,
                dtype=torch.float32
            )
        )

        self.nodes_gate_keys =  torch.nn.Linear(
            self.hidden_dim,
            num_nodes * self.num_heads
        )

        self.node_gate_values = torch.nn.Parameter(
            torch.randn(
                (
                    self.num_nodes,
                    self.num_heads,
                    self.hidden_dim,
                )
            ),
            requires_grad=True
        )

    def forward(self, x: RouteTensor) -> RouteTensor:
        x, node_items, head_items = self._match_gates(x)
        x, node_ind = self._route_tensors(x, node_items, head_items)
        return x, node_ind
    
    def _match_gates(
            self,
            x: RouteTensor
        ):
        """_match_gates takes the input tensors and matches them to the
        appropriate node and head. This is done by applying a linear
        transformation to the input tensors and then applying a softmax
        function to the result. The softmax function is applied to each
        head separately.

        Args:
            x (RouteTensor): Input tensors to be routed, shapes 
                ((l, h, d), ...) where l is the sequence length/node count
                h is the number of heads, and d is the hidden dimension.
                The tensors are grouped by node.
        
        Returns:
            x (RouteTensor): The input tensors multiplied by the gate
                values.
            node_items (RouteTensor): The node index of each input tensor.
            head_items (RouteTensor): The head index of each input tensor.
        """

        v = x.apply(self.nodes_gate_keys)
        v = v.apply(self.gate_dropout)
        x_items = []
        head_items = []
        node_items = []
        for item, x_item in zip(v.to_batches(), x.to_batches()):
            # TODO: apply softmax to each head separately
            # print(item.shape)

            val, ind = torch.topk(item, self.k, dim=-1)
            node_head = ind % self.num_heads
            node_ind = ind // self.num_heads
            head_items.append(node_head)
            node_items.append(node_ind)
            item = val[:, :, None] * x_item[:, None, :]
            x_items.append(item)

        x = RouteTensor(data=x_items, batch_inds=v.batch_inds)
        head_items = RouteTensor(data=head_items)
        node_items = RouteTensor(data=node_items)
        return x, node_items, head_items

    def _route_tensors(
            self,
            x: RouteTensor,
            node_ind: torch.Tensor,
            node_head: torch.Tensor
        ):
        """Route tensors essentially reorders the input tensors to be
        grouped by node. This is done by summing the tensors that are
        routed to the same node. This is done for each head separately.

        Args:
            x (RouteTensor): Input tensors to be routed, shapes 
                ((l, h, d), ...) where l is the sequence length/node count
                h is the number of heads/gates, and d is the hidden dimension.
            node_ind (torch.Tensor): The node index for each tensor in x.
                This corresponds to the node that the tensor is routed to.
            node_head (torch.Tensor): The head index for each tensor in x.
                This corresponds to the head/gate on the node that the tensor
                is routed to.

        Returns:
            RouteTensor: The routed tensors, shapes ((l, d), ...) where l is
                the sequence length/node count and d is the hidden dimension. 
                These tensors are reordered to be grouped by node.
            torch.Tensor: The node index for each tensor in the output. i.e.
                the order of the tensors in the output.
        """

        # add gate embeddings to x
        gates = self.node_gate_values[node_ind.data, node_head.data]
        x.data = x.data * gates
        x = x.apply(self.node_dropout)

        # reroute to nodes
        new_x = []
        node_inds_batch = []
        for ind, bi in enumerate(x.to_batches()):
            bx = []
            node_inds = []
            for ni in range(self.num_nodes):
                bool_inds = (node_ind[ind] == ni)
                # possible that no values are routed to this node
                if not bool_inds.any():
                    continue
                a = bi[bool_inds] # (h, d) per node
                a = a.sum(dim=0) # Sum over input heads
                bx.append(a)
                node_inds.append(ni)
            new_x.append(torch.stack(bx, dim=0))
            node_inds_batch.append(torch.tensor(node_inds))

        x = RouteTensor(data=new_x)
        node_ind = RouteTensor(data=node_inds_batch)
        return x, node_ind
