import math
import torch
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian, remove_self_loops


class ChebIIConv(MessagePassing):
    """
    One layer operation of ChebNetII:
    "Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited".

    Key definitions:
      - L = I - D^{-1/2} A D^{-1/2}  (normalized Laplacian)
      - L_tilde = 2/Lambda_max * L - I  (scaled+shifted Laplacian, spectrum in [-1, 1])
      - T_0(x)=x, T_1(x)=L_tilde x, T_k(x)=2 L_tilde T_{k-1}(x) - T_{k-2}(x)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        lambda_max: float = 2.0,
        cached: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = int(K)
        self.lambda_max = float(lambda_max)
        self.cached = bool(cached)
        self.linear = Linear(in_channels, out_channels, bias=bias)

        # Learnable interpolation values at Chebyshev nodes
        self.gamma = Parameter(torch.randn(self.K + 1))

        # Precompute Chebyshev nodes x_j and T_eval matrix (for coeff construction)
        j = torch.arange(self.K + 1, dtype=torch.float)
        nodes = torch.cos(math.pi * (j + 0.5) / (self.K + 1))
        T = torch.zeros(self.K + 1, self.K + 1)
        T[0] = 1
        if self.K >= 1:
            T[1] = nodes
        for k in range(2, self.K + 1):
            T[k] = 2 * nodes * T[k - 1] - T[k - 2]

        self.register_buffer("nodes", nodes)
        self.register_buffer("T_eval", T)
        self._cached_lap = None  # (lap_index, lap_weight)

    def reset_parameters(self):
        self.gamma.data.uniform_(-1, 1)
        self.linear.reset_parameters()
        self._cached_lap = None

    def _needs_rebuild_cache(self, edge_index: Tensor) -> bool:
        if self._cached_lap is None or not self.cached:
            return True
        cached_edge_index, cached_edge_weight = self._cached_lap
        return cached_edge_index.device != edge_index.device or cached_edge_weight.device != edge_index.device

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor = None,
    ) -> Tensor:
        # default edge weights: unweighted graph
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        # ChebNet/ChebNetII typically assumes adjacency without self-loops.
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # Build (and optionally cache) scaled+shifted Laplacian L_tilde
        if self._needs_rebuild_cache(edge_index):
            lap_index, lap_weight = get_laplacian(edge_index, edge_weight, normalization="sym")
            lap_weight = lap_weight.to(x.dtype)

            # L_hat = (2/lambda_max)*L - I
            lap_weight = lap_weight * (2.0 / self.lambda_max)
            mask = lap_index[0] == lap_index[1]
            lap_weight[mask] = lap_weight[mask] - 1.0

            self._cached_lap = (lap_index, lap_weight)
        else:
            lap_index, lap_weight = self._cached_lap

        # Chebyshev coefficients a_k
        Np1 = self.K + 1
        a = []
        for k in range(Np1):
            coef = (1.0 / Np1) if k == 0 else (2.0 / Np1)
            a_k = coef * (self.gamma * self.T_eval[k].to(self.gamma.dtype)).sum()
            a.append(a_k)
        a = torch.stack(a).to(x.dtype)

        # Recursive Chebyshev propagation
        out = a[0] * x
        if self.K >= 1:
            Tkm2 = x                          # T_0
            Tkm1 = self.propagate(lap_index, x=x, norm=lap_weight)  # T_1 = L_tilde x
            out = out + a[1] * Tkm1

            for k in range(2, Np1):
                Tk = 2.0 * self.propagate(lap_index, x=Tkm1, norm=lap_weight) - Tkm2
                out = out + a[k] * Tk
                Tkm2, Tkm1 = Tkm1, Tk

        return self.linear(out)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, K={self.K})"
        )
