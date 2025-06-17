import torch
from torch.nn import Parameter, Linear
from typing import Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GPRConv(MessagePassing):
    r"""Generalized PageRank convolution with feature projection.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
        K (int, optional): Number of propagation steps. Default is 10.
        alpha (float, optional): Initialization hyperparameter (use int for 'SGC'). Default is 0.1.
        Init (str, optional): Initialization scheme, one of ['SGC', 'PPR', 'NPPR', 'Random', 'WS']. Defaults to 'PPR'.
        Gamma (list or Tensor, optional): Pre-specified weights for 'WS' init. Default is None.
        bias (bool, optional): If False, the layer will not learn an additive bias. Default is True.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        Init: str = 'PPR',
        Gamma: Optional[torch.Tensor] = None,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        # feature projection
        self.lin = Linear(in_channels, out_channels, bias=bias)

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS'], \
            f"Unsupported Init mode '{Init}'"

        # Initialize propagation coefficients
        if Init == 'SGC':
            TEMP = torch.zeros(self.K + 1, dtype=torch.float)
            idx = int(self.alpha)
            if idx < 0 or idx > self.K:
                raise ValueError(f"alpha (SGC) must be integer in [0, K], got {self.alpha}")
            TEMP[idx] = 1.0
        elif Init == 'PPR':
            arange = torch.arange(self.K + 1, dtype=torch.float)
            TEMP = self.alpha * (1 - self.alpha) ** arange
            TEMP[-1] = (1 - self.alpha) ** self.K
        elif Init == 'NPPR':
            arange = torch.arange(self.K + 1, dtype=torch.float)
            TEMP = self.alpha ** arange
            TEMP = TEMP / TEMP.abs().sum()
        elif Init == 'Random':
            bound = (3.0 / (self.K + 1)) ** 0.5
            TEMP = torch.empty(self.K + 1).uniform_(-bound, bound)
            TEMP = TEMP / TEMP.abs().sum()
        else:  # 'WS'
            if Gamma is None:
                raise ValueError("Gamma must be provided for 'WS' initialization.")
            TEMP = torch.tensor(Gamma, dtype=torch.float)

        self.temp = Parameter(TEMP)

    def reset_parameters(self):
        """Reinitialize parameters and coefficients according to Init scheme."""
        self.lin.reset_parameters()
        with torch.no_grad():
            if self.Init == 'SGC':
                self.temp.zero_()
                self.temp[int(self.alpha)] = 1.0
            elif self.Init == 'PPR':
                arange = torch.arange(self.K + 1, dtype=self.temp.dtype, device=self.temp.device)
                temp = self.alpha * (1 - self.alpha) ** arange
                temp[-1] = (1 - self.alpha) ** self.K
                self.temp.copy_(temp)
            elif self.Init == 'NPPR':
                arange = torch.arange(self.K + 1, dtype=self.temp.dtype, device=self.temp.device)
                temp = self.alpha ** arange
                temp = temp / temp.abs().sum()
                self.temp.copy_(temp)
            elif self.Init == 'Random':
                bound = (3.0 / (self.K + 1)) ** 0.5
                temp = torch.empty(self.K + 1, dtype=self.temp.dtype, device=self.temp.device).uniform_(-bound, bound)
                temp = temp / temp.abs().sum()
                self.temp.copy_(temp)
            else:  # 'WS'
                gamma = torch.tensor(self.Gamma, dtype=self.temp.dtype, device=self.temp.device)
                self.temp.copy_(gamma)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        # Compute normalized adjacency (Â = D^{-1/2}(A+I)D^{-1/2})
        N = x.size(0) if num_nodes is None else num_nodes
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=N, dtype=x.dtype)

        # 0-hop term
        out = self.temp[0] * x
        h = x
        # propagate for k hops
        for k in range(self.K):
            h = self.propagate(edge_index, x=h, norm=norm)
            out = out + self.temp[k + 1] * h

        # project features to out_channels
        return self.lin(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}('  \
                f'in_channels={self.lin.in_features}, '  \
                f'out_channels={self.lin.out_features}, '  \
                f'K={self.K}, Init={self.Init}, alpha={self.alpha})')
