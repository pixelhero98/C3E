import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, SSGConv
from chebiiconv import ChebIIConv
from gprconv import GPRConv
from jacobiconv import JACOBIConv
from appnpconv import APPNPConv


class Model(nn.Module):
    def __init__(
        self,
        prop_layer,
        num_class,
        drop_probs,
        use_activations=None,
        conv_methods=None
    ):
        super().__init__()
        # Number of graph-conv layers
        self.num_layers = len(prop_layer) - 1

        # Determine convolution method(s)
        # conv_methods can be: None (default GCN), a str, or an iterable of str
        default = 'gcn'
        if conv_methods is None:
            methods = [default] * self.num_layers
        elif isinstance(conv_methods, str):
            methods = [conv_methods] * self.num_layers
        else:
            if len(conv_methods) != self.num_layers:
                raise ValueError(
                    f"conv_methods length must be {self.num_layers}, got {len(conv_methods)}"
                )
            methods = list(conv_methods)

        # Map string key to convolution class
        conv_map = {
            'gcn': GCNConv,
            'appnp': APPNPConv,
            'gdc_ppr': GCNConv,
            'gdc_hk': GCNConv,
            'sgc': SGConv,
            's2gc': SSGConv,
            'jacobi': JACOBIConv,
            'gprgnn': GPRConv,
            'chebnetii': ChebIIConv
        }

        # Graph convolution layers
        self.propagation = nn.ModuleList()
        for i, method in enumerate(methods):
            key = method.lower()
            if key not in conv_map:
                raise ValueError(f"Unknown conv method '{method}' at layer {i}")
            Conv = conv_map[key]
            # GATConv needs heads; default to 1
            if key == 'jacobi':
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], K=3, aggr='gcn', a=1.0, b=1.0, cached=True)
                )
            elif key == 'gprgnn':
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], K=10, alpha=0.1, Init='PPR', Gamma=None, bias=True)
                )
            elif key == 'chebnetii':
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], K=1, lambda_max=2.0, cached=True, bias=True)
                )
            elif key == 'appnp':
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], K=10, alpha=0.1, dropout=0， cached=False, add_self_loops=True, normalize=True, bias=True)
                )
            elif key == 'sgc':
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], K=1, cached=False, add_self_loops=True, bias=True)
                )
            elif key == 's2gc':
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], alpha=0.1, K=1, cached=False, add_self_loops=True, bias=True)
                )
            else:
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1])
                )

        # Activation functions
        self.activations = nn.ModuleList(
            [nn.PReLU(num_parameters=prop_layer[i + 1]) for i in range(self.num_layers)]
        )

        # Flags to enable/disable activation per layer
        if use_activations is None:
            self.use_activations = [True] * self.num_layers
        elif isinstance(use_activations, bool):
            self.use_activations = [use_activations] * self.num_layers
        else:
            if len(use_activations) != self.num_layers:
                raise ValueError(
                    f"use_activations length must be {self.num_layers}, got {len(use_activations)}"
                )
            self.use_activations = list(use_activations)

        # Residual (linear) projections for layers 1..L-1
        self.residuals = nn.ModuleList(
            [nn.Linear(prop_layer[i], prop_layer[i + 1]) for i in range(1, self.num_layers)]
        )

        # Dropout modules stored at initialization
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dp) for dp in drop_probs]
        )

        # Final classifier
        self.classifier = nn.Linear(prop_layer[-1], num_class)

    def forward(self, data):
        """
        x:           Node feature matrix of shape [N, prop_layer[0]]
        edge_index: Graph connectivity in COO format
        """
        h = data.x
        for i, conv in enumerate(self.propagation):
            # Graph convolution
            h_new = conv(h, data.edge_index)

            # Add residual for layers beyond the first
            if i > 0:
                res = self.residuals[i-1](h)
                h_new = h_new + res

            # Apply activation if enabled
            if self.use_activations[i]:
                h_new = self.activations[i](h_new)

            # Apply dropout
            h = self.dropouts[i](h_new)

        # Classification head
        out = self.classifier(h)
        return out
