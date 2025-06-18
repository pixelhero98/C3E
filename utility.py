import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor

# Ensure your model and data live on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_rep(model, data):
    """
    Run a forward pass and collect per-layer activations via hooks.
    Collects outputs of each propagation (conv) and activation layer.
    All outputs are detached and moved to CPU immediately to avoid GPU memory bloat.
    """
    model = model.to(device)
    data = data.to(device)
    reps = []
    hooks = []

    # Hook functions to capture outputs
    def hook_conv(module, input, output):
        reps.append(output.detach().cpu())

    def hook_act(module, input, output):
        reps.append(output.detach().cpu())

    # Register hooks on propagation (conv) layers
    for conv in model.propagation:
        hooks.append(conv.register_forward_hook(hook_conv))
    # Register hooks on activation layers
    for act in model.activations:
        hooks.append(act.register_forward_hook(hook_act))

    # Forward pass
    try:
        model(data)
    finally:
        for h in hooks:
            h.remove()

    return reps


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

    Example for your Model class:
        model = load_model(
            Model,
            "checkpoint.pth",
            prop_layer=[in_dim, hidden1, ..., out_dim],
            num_class=dataset.num_classes,
            drop_probs=[0.5, 0.5, ...],
            use_activations=[True, True, ...],
            conv_methods=[...]
        )
    """
    model = model_class(*model_args, **model_kwargs)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Additional Laplacian utilities
# -----------------------------------------------------------------------------

def compute_normalized_laplacian(edge_index, num_nodes: int) -> SparseTensor:
    """
    Build the symmetric normalized Laplacian:
      L_sym = I - D^{-1/2} A D^{-1/2}
    This differs from the unnormalized Laplacian L = D - A in that "mass"
    from high-degree nodes is downweighted, yielding a spectrum in [0,2]
    and better numerical stability for many spectral methods.
    """
    row, col = edge_index
    N = num_nodes
    # Adjacency
    A = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    # Degree diagonal
    deg = A.sum(dim=1)
    # Inverse sqrt degree
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = SparseTensor(
        row=torch.arange(N, device=deg.device),
        col=torch.arange(N, device=deg.device),
        value=deg_inv_sqrt
    )
    # Identity
    I = SparseTensor.eye(N, device=deg.device)
    # Symmetric normalized Laplacian
    L_sym = I - D_inv_sqrt.matmul(A).matmul(D_inv_sqrt)
    return L_sym

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
