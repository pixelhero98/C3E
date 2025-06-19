import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor, spmm, to_edge_index
from torch_geometric.utils.sparse import set_sparse_value
from torch.nn import Linear

class APPNPConv(MessagePassing):
    r"""Approximate Personalized Propagation of Neural Predictions convolution
    layer with input/output feature transformation.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Number of propagation iterations. Default: 10.
        alpha (float): Teleport probability. Default: 0.1.
        dropout (float, optional): Dropout probability for edges. Default: 0.
        cached (bool, optional): Whether to cache the normalized adjacency. Default: False.
        add_self_loops (bool, optional): Whether to add self-loops. Default: True.
        normalize (bool, optional): Whether to apply symmetric normalization. Default: True.
        bias (bool, optional): If False, the layer will not learn an additive bias. Default: True.
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.lin = Linear(in_channels, out_channels, bias=bias)

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: Optional[int] = None
    ) -> Tensor:
        # Initial embedding
        h0 = x

        # Normalize and cache adjacency
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight,
                        num_nodes or x.size(self.node_dim),
                        False, self.add_self_loops,
                        self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache
            else:
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight,
                        num_nodes or x.size(self.node_dim),
                        False, self.add_self_loops,
                        self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = h0
        for _ in range(self.K):
            # edge dropout
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    if is_torch_sparse_tensor(edge_index):
                        _, ew = to_edge_index(edge_index)
                        ew = F.dropout(ew, p=self.dropout)
                        edge_index = set_sparse_value(edge_index, ew)
                    else:
                        assert edge_weight is not None
                        edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    val = edge_index.storage.value()
                    assert val is not None
                    val = F.dropout(val, p=self.dropout)
                    edge_index = edge_index.set_value(val, layout='coo')

            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            x = (1 - self.alpha) * x + self.alpha * h0

        # Final linear transformation
        return self.lin(x)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('  \
                f'in_channels={self.in_channels}, '  \
                f'out_channels={self.out_channels}, '  \
                f'K={self.K}, alpha={self.alpha})')
