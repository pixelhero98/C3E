import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree
from torch_sparse import SparseTensor

def buildAdj(edge_index: Tensor,
             edge_weight: Tensor,
             n_node: int,
             aggr: str = "gcn") -> SparseTensor:
    """
    Constructs (normalized) SparseTensor adjacency exactly as in impl/PolyConv.py.
    Supports 'mean', 'sum', 'gcn'; moves to CUDA if needed.
    """
    deg = degree(edge_index[0], n_node)
    deg[deg < 0.5] += 1.0

    if aggr == "mean":
        val = (1.0 / deg)[edge_index[0]] * edge_weight
    elif aggr == "sum":
        val = edge_weight
    elif aggr == "gcn":
        deg_inv_sqrt = deg.pow(-0.5)
        val = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]
    else:
        raise NotImplementedError(f"Unknown aggr: {aggr}")

    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=val,
                       sparse_sizes=(n_node, n_node)).coalesce()
    return adj.cuda() if edge_index.is_cuda else adj


def JacobiConv(L: int,
               xs: list[Tensor],
               adj: SparseTensor,
               alphas: list[Tensor],
               a: float = 1.0,
               b: float = 1.0,
               l: float = -1.0,
               r: float = 1.0) -> Tensor:
    """
    Implements the JacobiConv recurrence from GraphPKU's PolyConv:
      • L=0: returns xs[0]
      • L=1: closed‐form with domain [l,r]
      • L>=2: three‐term recurrence with domain scaling
    """
    if L == 0:
        return xs[0]

    Ax = adj @ xs[-1]
    if L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * ((l + r) / (r - l))
        coef2 = (a + b + 2) / (r - l)
        return alphas[0] * (coef1 * xs[-1] + coef2 * Ax)

    coef_l      = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1  = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2  = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2    = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)

    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)

    tmp1_2 = tmp1 * (2.0 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2

    return tmp1_2 * Ax \
         - tmp2_2 * xs[-1] \
         - tmp3   * xs[-2]

class JACOBIConv(nn.Module):
    """
    One layer of PolyConvFrame with JacobiConv:
      - builds [P0 x, P1 x, … P_K x]
      - concatenates and a final linear projection
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 K: int = 3,
                 aggr: str = 'gcn',
                 a: float = 1.0,
                 b: float = 1.0,
                 cached: bool = True):
        super().__init__()
        self.K      = K
        self.aggr   = aggr
        self.a      = a
        self.b      = b
        self.cached = cached

        # base scale for alphas (tanh applied in forward)
        self.basealpha = 1.0
        self.alphas    = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0))
            for _ in range(K+1)
        ])

        self.lin  = nn.Linear((K+1)*in_channels, out_channels)
        self.adj  = None

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Tensor = None) -> Tensor:
        # default to unweighted (all-ones) if not provided
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        N = x.size(0)
        if self.adj is None or not self.cached:
            self.adj = buildAdj(edge_index, edge_weight, N, self.aggr)

        # apply tanh for stability, as in the original repo
        alphas = [self.basealpha * torch.tanh(a) for a in self.alphas]

        # compute P_0 x, P_1 x, ..., P_K x
        xs = [x]
        for L in range(1, self.K+1):
            xs.append(JacobiConv(L, xs, self.adj, alphas, self.a, self.b))

        # concatenate and linear project
        x_cat = torch.cat(xs, dim=1)
        return self.lin(x_cat)
