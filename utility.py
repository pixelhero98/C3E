import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor

# Ensure your model and data live on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_rep(model, data, p):
    """
    Run a forward pass and collect per-layer activations via hooks.
    All outputs are detached and moved to CPU immediately to avoid GPU memory bloat.
    """
    model = model.to(device)
    data = data.to(device)
    ops = []

    # Choose layers explicitly; skip the first mmp_layer since activation1 covers it
    layers = [model.activation1] + model.mmp_layer[1:] + [model.postpro_layer[0]]
    hooks = []

    def make_hook(idx):
        def hook(module, input, output):
            ops.append(output.detach().cpu())
        return hook

    # Register hooks
    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Forward pass
    try:
        model(data.x, data.edge_index, p)
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()

    return ops


def rep_entropy(rep, nbins: int = 100) -> float:
    """
    Estimate Shannon entropy of a flattened tensor by:
      1. Softmax to map values into [0,1]
      2. Histogram into `nbins` bins over [0,1]
      3. Compute entropy = -sum(p_i log p_i)
    """
    rep_flat = rep.view(-1)
    p = F.softmax(rep_flat, dim=0)

    # Define bin edges and assign each p to a bin
    edges = torch.linspace(0.0, 1.0, steps=nbins + 1, device=p.device)
    bin_idxs = torch.bucketize(p, edges, right=False) - 1

    # Count frequencies
    counts = torch.bincount(bin_idxs, minlength=nbins).float()
    probs = counts / counts.sum()

    # Numerical stability
    eps = 1e-12
    entropy = -(probs * torch.log(probs + eps)).sum().item()
    return entropy


def show_layer_rep_entropy(ops, data, nbins: int = 100):
    """
    Print normalized entropy for input features and each collected layer.
    """
    data = data.to(device)

    entropies = {0: rep_entropy(data.x, nbins)}
    for idx, rep in enumerate(ops, start=1):
        entropies[idx] = rep_entropy(rep, nbins)

    print(entropies)
    print('===============================================================')


def compute_sparse_laplacian(edge_index, num_nodes: int) -> SparseTensor:
    """
    Build the unnormalized graph Laplacian as a SparseTensor.
    L = D - A
    """
    row, col = edge_index
    N = num_nodes
    # Adjacency
    A = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    # Degree diagonal
    deg = A.sum(dim=1)
    D = SparseTensor(row=torch.arange(N, device=deg.device),
                     col=torch.arange(N, device=deg.device),
                     value=deg)
    L = D - A
    return L


def compute_dirichlet_energy(node_features: torch.Tensor, L: SparseTensor) -> float:
    """
    Compute Dirichlet energy: trace(X^T L X).
    Works with sparse Laplacian.
    """
    X = node_features.to(device)
    # LX = L * X
    LX = L.matmul(X)
    energy = torch.trace(X.t() @ LX)
    return energy.item()


def show_layer_dirichlet_energy(ops, data):
    """
    Print normalized Dirichlet energy for input and each representation.
    """
    data = data.to(device)
    L = compute_sparse_laplacian(data.edge_index, data.x.size(0))

    # Baseline energy of input
    base_energy = compute_dirichlet_energy(data.x, L)
    energies = {0: 1.0}

    for idx, rep in enumerate(ops, start=1):
        energies[idx] = compute_dirichlet_energy(rep, L) / base_energy

    print(energies)
    print('===============================================================')

# Statistics: mean & variance per dimension
def vector_mean(rep: torch.Tensor) -> torch.Tensor:
    """Mean over the node dimension."""
    return rep.mean(dim=0)

def vector_variance(rep: torch.Tensor) -> torch.Tensor:
    """Variance (unbiased) over the node dimension."""
    return rep.var(dim=0, unbiased=True)

# Aggregations
def mean_activation_per_layer(ops):
    """
    Compute the global mean activation per layer (flattened).
    """
    return [rep.mean().item() for rep in ops]


def mean_node_activation_per_layer(ops):
    """
    Compute per-layer mean activation averaged over nodes (vector of size = hidden_dim).
    """
    return [vector_mean(rep).cpu() for rep in ops]

# -----------------------------------------------------------------------------
# Model persistence utilities
# -----------------------------------------------------------------------------

def save_model(model: torch.nn.Module, path: str):
    """
    Save the model's state_dict to the given file path.
    """
    torch.save(model.state_dict(), path)


def load_model(model_class, path: str, *model_args, **model_kwargs) -> torch.nn.Module:
    """
    Instantiate a model from model_class with provided args/kwargs,
    load its state_dict from path, move it to the correct device,
    and set to evaluation mode.

    Example:
        model = load_model(MyGNN, "checkpoint.pth", in_channels, hidden_channels, out_channels)
    """
    model = model_class(*model_args, **model_kwargs)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Usage example (in a separate script)
# -----------------------------------------------------------------------------
# Example showing how to save, reload with `p`, and compute metrics
#
# from improved_graph_metrics import (
#     save_model, load_model,
#     get_model_rep, show_layer_rep_entropy, show_layer_dirichlet_energy
# )
# from my_model import MyGNNModel
# from torch_geometric.datasets import Planetoid
#
# # Load data
# dataset = Planetoid("/data/Cora", "Cora")  # dataset[0]
# data = dataset[0]
#
# # After training, save the model state:
# #   save_model(trained_model, "checkpoint.pth")
#
# # Later: reload the model (including `p` if model __init__ requires it)
# model = load_model(
#     MyGNNModel,             # model class
#     "checkpoint.pth",      # saved state file
#     in_channels=dataset.num_node_features,
#     hidden_channels=64,
#     out_channels=dataset.num_classes,
#     p=0.5                   # pass `p` into constructor if needed
# )
#
# # Collect activations; forward still takes `p`
# reps = get_model_rep(model, data, p=0.5)
#
# # Compute and display metrics
# show_layer_rep_entropy(reps, data)
# show_layer_dirichlet_energy(reps, data)
