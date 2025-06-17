import math
import torch
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian

class ChebNetIIConv(MessagePassing):
    """
    ChebNetII convolution layer from
    "Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited".

    Approximates spectral convolutions via Chebyshev interpolation:
        P_N(x) = sum_{k=0}^N a_k T_k(x),
    with a_k = (2/(N+1)) sum_{j=0}^N gamma_j T_k(x_j) (except a_0 uses 1/(N+1)).

    Args:
        in_channels (int): Dimension of input features.
        out_channels (int): Dimension of output features.
        K (int): Order of Chebyshev polynomials.
        lambda_max (float): Largest eigenvalue of normalized Laplacian (default 2.0).
        cached (bool): Whether to cache the Laplacian (default True).
        bias (bool): If set to False, the layer will not learn an additive bias.
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

        # precompute Chebyshev nodes x_j = cos((j+0.5)pi/(K+1))
        j = torch.arange(K + 1, dtype=torch.float)
        nodes = torch.cos(math.pi * (j + 0.5) / (K + 1))

        # precompute T_k(x_j) matrix via recurrence
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
        """
        Args:
            x (Tensor): Node feature matrix [N, in_channels].
            edge_index (Tensor): Edge indices [2, E].
            edge_weight (Tensor, optional): Edge weights [E].

        Returns:
            Tensor: Updated node features [N, out_channels].
        """
        # Compute (and cache) normalized Laplacian L = get_laplacian
        if self._cached_lap is None or not self.cached:
            lap_index, lap_weight = get_laplacian(
                edge_index, edge_weight, normalization='sym')
            # scale by 2/lambda_max
            lap_weight = lap_weight * (2.0 / self.lambda_max)
            self._cached_lap = (lap_index, lap_weight)
        else:
            lap_index, lap_weight = self._cached_lap

        # Compute Chebyshev coefficients a_k from gamma and T_eval
        # a_0 = (1/(K+1)) * sum_j gamma_j * T_0(x_j)
        # a_k = (2/(K+1)) * sum_j gamma_j * T_k(x_j), for k >= 1
        Np1 = self.K + 1
        a = []
        for k in range(Np1):
            coef = 1.0 / Np1 if k == 0 else 2.0 / Np1
            a_k = coef * (self.gamma * self.T_eval[k]).sum()
            a.append(a_k)
        a = torch.stack(a)

        # Recursive Chebyshev propagation
        out = a[0] * x
        if self.K >= 1:
            h = self.propagate(lap_index, x=x, norm=lap_weight)
            out = out + a[1] * h

        Tkm2, Tkm1 = x, h if self.K >= 1 else (x, x)
        for k in range(2, Np1):
            Tkm2, Tkm1 = Tkm1, 2 * self.propagate(lap_index, x=Tkm1, norm=lap_weight) - Tkm2
            out = out + a[k] * Tkm1

        return self.linear(out)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        # Message passing: multiply neighbor features by norm scalar
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, K={self.K})'
