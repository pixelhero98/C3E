import math
import torch
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian


class ChebIIConv(MessagePassing):
    """
    One layer operation of ChebNetII:
    "Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited".
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 K: int,
                 lambda_max: float = 2.0,
                 cached: bool = True,
                 bias: bool = True):
        super().__init__(aggr='add')  # sum aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.lambda_max = lambda_max
        self.cached = cached
        self.linear = Linear(in_channels, out_channels, bias=bias)

        # learnable interpolation values at Chebyshev nodes
        self.gamma = Parameter(torch.randn(K + 1))

        # precompute Chebyshev nodes x_j and T_eval matrix
        j = torch.arange(K + 1, dtype=torch.float)
        nodes = torch.cos(math.pi * (j + 0.5) / (K + 1))
        T = torch.zeros(K + 1, K + 1)
        T[0] = 1
        if K >= 1:
            T[1] = nodes
        for k in range(2, K + 1):
            T[k] = 2 * nodes * T[k - 1] - T[k - 2]
        self.register_buffer('nodes', nodes)
        self.register_buffer('T_eval', T)
        self._cached_lap = None

    def reset_parameters(self):
        self.gamma.data.uniform_(-1, 1)
        self.linear.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Tensor = None) -> Tensor:
        # build edge_weight default
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        # Compute (and cache) normalized Laplacian L and then shift to [-1,1]
        if self._cached_lap is None or not self.cached:
            lap_index, lap_weight = get_laplacian(
                edge_index, edge_weight, normalization='sym')
            # scale by 2/lambda_max
            lap_weight = lap_weight * (2.0 / self.lambda_max)
            # subtract identity to shift eigenvalues to [-1,1]
            mask = lap_index[0] == lap_index[1]
            lap_weight[mask] = lap_weight[mask] - 1.0
            self._cached_lap = (lap_index, lap_weight)
        else:
            lap_index, lap_weight = self._cached_lap

        # compute Chebyshev coefficients a_k
        Np1 = self.K + 1
        a = []
        for k in range(Np1):
            coef = 1.0 / Np1 if k == 0 else 2.0 / Np1
            a_k = coef * (self.gamma * self.T_eval[k]).sum()
            a.append(a_k)
        a = torch.stack(a)

        # recursive Chebyshev propagation
        out = a[0] * x
        if self.K >= 1:
            h = self.propagate(lap_index, x=x, norm=lap_weight)
            # apply shift: propagate gives ((2/Lmax)L) x, so subtract x
            h = h - x
            out = out + a[1] * h
        Tkm2, Tkm1 = x, h if self.K >= 1 else (x, x)
        for k in range(2, Np1):
            Tkm2, Tkm1 = Tkm1, 2 * (self.propagate(lap_index, x=Tkm1, norm=lap_weight) - Tkm2) - Tkm2
            out = out + a[k] * Tkm1

        return self.linear(out)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(' \
                f'in_channels={self.in_channels}, ' \
                f'out_channels={self.out_channels}, ' \
                f'K={self.K})')
