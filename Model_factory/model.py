import torch
import torch.nn as nn
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
        conv_methods=None,
    ):
        super().__init__()

        # Number of graph-conv layers
        self.num_layers = len(prop_layer) - 1
        if self.num_layers <= 0:
            raise ValueError(f"prop_layer must have length >= 2, got {len(prop_layer)}")

        # Validate dropout schedule length early (prevents forward-time IndexError).
        if len(drop_probs) != self.num_layers:
            raise ValueError(
                f"drop_probs length must be {self.num_layers}, got {len(drop_probs)}"
            )

        # Determine convolution method(s)
        default = "gcn"
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

        conv_map = {
            "gcn": GCNConv,
            "appnp": APPNPConv,
            "gdc": GCNConv,      # if you later apply GDC diffusion, pass weights via data.edge_attr/edge_weight
            "sgc": SGConv,
            "s2gc": SSGConv,
            "jacobi": JACOBIConv,
            "jacobiconv": JACOBIConv,
            "gprgnn": GPRConv,
            "chebnetii": ChebIIConv,
        }

        def _try_construct(Cls, *args, **kwargs):
            """Construct layer with kwargs if supported; otherwise fall back to no-kwargs."""
            try:
                return Cls(*args, **kwargs)
            except TypeError:
                # Backward-compat for older PyG signatures.
                kwargs.pop("add_self_loops", None)
                return Cls(*args, **kwargs)

        # Propagation layers
        self.propagation = nn.ModuleList()
        for i, method in enumerate(methods):
            key = str(method).lower()
            if key not in conv_map:
                raise ValueError(f"Unknown conv method '{method}' at layer {i}")
            Conv = conv_map[key]

            if key in {"jacobi", "jacobiconv"}:
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], K=3, aggr="gcn", a=1.0, b=1.0)
                )
            elif key == "gprgnn":
                self.propagation.append(Conv(prop_layer[i], prop_layer[i + 1], K=10, alpha=0.1))
            elif key == "chebnetii":
                self.propagation.append(Conv(prop_layer[i], prop_layer[i + 1], K=1, lambda_max=2.0))
            elif key == "appnp":
                # train_val_test currently applies T.AddSelfLoops(); avoid double looping.
                self.propagation.append(
                    Conv(prop_layer[i], prop_layer[i + 1], K=10, alpha=0.1, add_self_loops=False)
                )
            elif key == "sgc":
                self.propagation.append(
                    _try_construct(Conv, prop_layer[i], prop_layer[i + 1], K=1, add_self_loops=False)
                )
            elif key == "s2gc":
                self.propagation.append(
                    _try_construct(Conv, prop_layer[i], prop_layer[i + 1], alpha=0.1, K=1, add_self_loops=False)
                )
            else:
                # GCNConv: avoid double self-looping when the dataset transform already adds self-loops.
                if key in {"gcn", "gdc"}:
                    self.propagation.append(_try_construct(Conv, prop_layer[i], prop_layer[i + 1], add_self_loops=False))
                else:
                    self.propagation.append(Conv(prop_layer[i], prop_layer[i + 1]))

        self.activations = nn.ModuleList(
            [nn.PReLU(num_parameters=prop_layer[i + 1]) for i in range(self.num_layers)]
        )

        # Activation enable/disable flags
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

        # Residual projections for layers 1..L-1
        self.residuals = nn.ModuleList(
            [nn.Linear(prop_layer[i], prop_layer[i + 1]) for i in range(1, self.num_layers)]
        )

        self.dropouts = nn.ModuleList([nn.Dropout(p=float(dp)) for dp in drop_probs])
        self.classifier = nn.Linear(prop_layer[-1], num_class)

    def forward(self, data):
        h = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None:
            edge_weight = getattr(data, "edge_attr", None)

        for i, conv in enumerate(self.propagation):
            # Prefer passing edge weights when present; fall back if a layer doesn't accept it.
            try:
                h_new = conv(h, edge_index, edge_weight=edge_weight) if edge_weight is not None else conv(h, edge_index)
            except TypeError:
                h_new = conv(h, edge_index)

            if i > 0:
                h_new = h_new + self.residuals[i - 1](h)

            if self.use_activations[i]:
                h_new = self.activations[i](h_new)

            h = self.dropouts[i](h_new)

        return self.classifier(h)
