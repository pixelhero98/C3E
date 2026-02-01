import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor, spmm, to_edge_index
from torch_geometric.utils.sparse import set_sparse_value


class APPNPConv(MessagePassing):
    """One layer operation of APPNP with input/output feature transformation.

    This implementation avoids *double* self-loop addition:
      - If the input already contains self-loops, we do not add them again in gcn_norm.
      - Otherwise we add them once (when add_self_loops=True).
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.0,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = int(K)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.cached = bool(cached)
        self.add_self_loops = bool(add_self_loops)
        self.normalize = bool(normalize)
        self.lin = Linear(in_channels, out_channels, bias=bias)

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    @staticmethod
    def _tensor_has_self_loops(edge_index: Tensor) -> bool:
        if is_torch_sparse_tensor(edge_index):
            ei, _ = to_edge_index(edge_index)
            return bool((ei[0] == ei[1]).any().item())
        return bool((edge_index[0] == edge_index[1]).any().item())

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        # Initial embedding (before propagation)
        h0 = x

        # Normalize and cache adjacency (cache is device-sensitive)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                cache_valid = (
                    cache is not None
                    and cache[0].device == x.device
                    and (cache[1] is None or cache[1].device == x.device)
                )
                if not cache_valid:
                    add_loops = self.add_self_loops and (not self._tensor_has_self_loops(edge_index))
                    edge_index, edge_weight = gcn_norm(
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        num_nodes=num_nodes or x.size(self.node_dim),
                        improved=False,
                        add_self_loops=add_loops,
                        flow=self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache
            else:
                cache = self._cached_adj_t
                cache_valid = cache is not None and getattr(cache, "device", None) == x.device
                if not cache_valid:
                    # For SparseTensor inputs we can't cheaply check for existing loops;
                    # we respect self.add_self_loops as-is.
                    edge_index = gcn_norm(
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        num_nodes=num_nodes or x.size(self.node_dim),
                        improved=False,
                        add_self_loops=self.add_self_loops,
                        flow=self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = h0
        for _ in range(self.K):
            # Edge dropout
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    if is_torch_sparse_tensor(edge_index):
                        _, ew = to_edge_index(edge_index)
                        ew = F.dropout(ew, p=self.dropout, training=True)
                        edge_index = set_sparse_value(edge_index, ew)
                    else:
                        assert edge_weight is not None
                        edge_weight = F.dropout(edge_weight, p=self.dropout, training=True)
                else:
                    val = edge_index.storage.value()
                    assert val is not None
                    val = F.dropout(val, p=self.dropout, training=True)
                    edge_index = edge_index.set_value(val, layout="coo")

            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            x = (1 - self.alpha) * x + self.alpha * h0

        # Final linear transformation
        return self.lin(x)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, K={self.K}, alpha={self.alpha})"
        )
