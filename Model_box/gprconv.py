import torch
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GPRConv(MessagePassing):
    r"""Generalized PageRank convolution from
    "Adaptive Universal Generalized PageRank Graph Neural Network".

    Args:
        K (int): Number of propagation steps.
        alpha (float): Initialization hyperparameter.
        Init (str): Initialization scheme, one of ['SGC', 'PPR', 'NPPR', 'Random', 'WS'].
        Gamma (array, optional): Pre-specified weights (for 'WS' init).
    """
    def __init__(self, K: int, alpha: float, Init: str = 'PPR', Gamma=None, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']

        # Initialize coefficients
        if Init == 'SGC':
            TEMP = torch.zeros(K + 1, dtype=torch.float)
            TEMP[int(alpha)] = 1.0
        elif Init == 'PPR':
            arange = torch.arange(K + 1, dtype=torch.float)
            TEMP = alpha * (1 - alpha) ** arange
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            arange = torch.arange(K + 1, dtype=torch.float)
            TEMP = alpha ** arange
            TEMP = TEMP / TEMP.abs().sum()
        elif Init == 'Random':
            bound = (3.0 / (K + 1)) ** 0.5
            TEMP = torch.empty(K + 1).uniform_(-bound, bound)
            TEMP = TEMP / TEMP.abs().sum()
        else:  # 'WS'
            TEMP = torch.tensor(Gamma, dtype=torch.float)

        self.temp = Parameter(TEMP)

    def reset_parameters(self):
        """Reinitialize coefficients according to Init scheme."""
        with torch.no_grad():
            if self.Init == 'SGC':
                self.temp.zero_()
                self.temp[int(self.alpha)] = 1.0
            elif self.Init == 'PPR':
                for k in range(self.K + 1):
                    self.temp[k] = self.alpha * (1 - self.alpha) ** k
                self.temp[-1] = (1 - self.alpha) ** self.K
            elif self.Init == 'NPPR':
                for k in range(self.K + 1):
                    self.temp[k] = self.alpha ** k
                self.temp[:] = self.temp / self.temp.abs().sum()
            elif self.Init == 'Random':
                bound = (3.0 / (self.K + 1)) ** 0.5
                self.temp.uniform_(-bound, bound)
                self.temp[:] = self.temp / self.temp.abs().sum()
            else:  # 'WS'
                self.temp[:] = torch.tensor(self.Gamma, dtype=self.temp.dtype)

    def forward(self, x, edge_index, edge_weight=None):
        # Compute normalized adjacency
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        # 0-hop term
        out = self.temp[0] * x
        h = x
        # propagate for k hops
        for k in range(self.K):
            h = self.propagate(edge_index, x=h, norm=norm)
            out = out + self.temp[k + 1] * h
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, Init={self.Init}, alpha={self.alpha})'
