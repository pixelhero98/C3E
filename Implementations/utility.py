import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor
import logging
from pathlib import Path
from torch import nn, optim
from torch.optim import lr_scheduler
from typing import Sequence


# Ensure your model and data live on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data, optimizer):
    """
    Perform one training step on the full graph.

    Args:
        model: torch.nn.Module
        data: PyG data object with x, edge_index, train_mask, y
        optimizer: torch optimizer

    Returns:
        float: training loss
    """
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def val(model, data):
    """
    Compute validation accuracy.

    Args:
        model: torch.nn.Module
        data: PyG data object with x, edge_index, val_mask, y

    Returns:
        float: validation accuracy
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = int((pred[data.val_mask] == data.y[data.val_mask]).sum())
        total = int(data.val_mask.sum())
    return correct / total


def test(model, data):
    """
    Compute test accuracy.

    Args:
        model: torch.nn.Module
        data: PyG data object with x, edge_index, test_mask, y

    Returns:
        float: test accuracy
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
        total = int(data.test_mask.sum())
    return correct / total
    
def get_model_rep(model, data):
    """
    Run a forward pass and collect per-layer outputs separately for
    propagation (conv) and activation layers.

    Returns:
        conv_reps: list of tensors from each conv layer (cpu, detached)
        act_reps: list of tensors from each activation layer (cpu, detached)
    """
    model = model.to(device)
    data = data.to(device)
    conv_reps = []
    act_reps = []
    hooks = []

    def hook_conv(module, input, output):
        conv_reps.append(output.detach().cpu())

    def hook_act(module, input, output):
        act_reps.append(output.detach().cpu())

    # Register hooks
    for conv in model.propagation:
        hooks.append(conv.register_forward_hook(hook_conv))
    for act in model.activations:
        hooks.append(act.register_forward_hook(hook_act))

    # Forward pass
    try:
        model(data)
    finally:
        for h in hooks:
            h.remove()

    return conv_reps, act_reps


def rep_entropy(rep, nbins: int = 100) -> float:
    """
    Estimate Shannon entropy of a flattened tensor by:
      1. Softmax to map values into [0,1]
      2. Histogram into `nbins` bins over [0,1]
      3. Compute entropy = -sum(p_i log p_i)
    """
    rep_flat = rep.view(-1)
    p = F.softmax(rep_flat, dim=0)

    edges = torch.linspace(0.0, 1.0, steps=nbins + 1, device=p.device)
    bin_idxs = torch.bucketize(p, edges, right=False) - 1
    counts = torch.bincount(bin_idxs, minlength=nbins).float()
    probs = counts / counts.sum()

    eps = 1e-12
    entropy = -(probs * torch.log(probs + eps)).sum().item()
    return entropy


def show_layer_rep_entropy(conv_reps, act_reps, data, nbins: int = 100):
    """
    Print entropy for input features and each conv and activation layer separately.
    """
    data = data.to(device)
    entropies = {'input': rep_entropy(data.x, nbins)}

    for idx, rep in enumerate(conv_reps, start=1):
        entropies[f'conv_{idx}'] = rep_entropy(rep, nbins)
    for idx, rep in enumerate(act_reps, start=1):
        entropies[f'act_{idx}'] = rep_entropy(rep, nbins)

    print(entropies)
    print('===============================================================')


def compute_sparse_laplacian(edge_index, num_nodes: int) -> SparseTensor:
    """
    Build the unnormalized graph Laplacian as a SparseTensor: L = D - A
    """
    row, col = edge_index
    N = num_nodes
    A = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = A.sum(dim=1)
    D = SparseTensor(
        row=torch.arange(N, device=deg.device),
        col=torch.arange(N, device=deg.device),
        value=deg
    )
    L = D - A
    return L


def compute_normalized_laplacian(edge_index, num_nodes: int) -> SparseTensor:
    """
    Build the symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    """
    row, col = edge_index
    N = num_nodes
    A = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = SparseTensor(
        row=torch.arange(N, device=deg.device),
        col=torch.arange(N, device=deg.device),
        value=deg_inv_sqrt
    )
    I = SparseTensor.eye(N, device=deg.device)
    L_sym = I - D_inv_sqrt.matmul(A).matmul(D_inv_sqrt)
    return L_sym


def compute_dirichlet_energy(node_features: torch.Tensor, L: SparseTensor) -> float:
    """
    Compute Dirichlet energy: trace(X^T L X).
    Works with both sparse and dense Laplacians.
    """
    X = node_features.to(device)
    LX = L.matmul(X) if hasattr(L, 'matmul') else L @ X
    energy = torch.trace(X.t() @ LX)
    return energy.item()


def show_layer_dirichlet_energy(conv_reps, act_reps, data):
    """
    Print normalized Dirichlet energy for input, conv, and activation layers.
    """
    data = data.to(device)
    L = compute_sparse_laplacian(data.edge_index, data.x.size(0))

    base_energy = compute_dirichlet_energy(data.x, L)
    energies = {'input': 1.0}
    for idx, rep in enumerate(conv_reps, start=1):
        energies[f'conv_{idx}'] = compute_dirichlet_energy(rep, L) / base_energy
    for idx, rep in enumerate(act_reps, start=1):
        energies[f'act_{idx}'] = compute_dirichlet_energy(rep, L) / base_energy

    print(energies)
    print('===============================================================')


def vector_mean(rep: torch.Tensor) -> torch.Tensor:
    """Mean over the node dimension."""
    return rep.mean(dim=0)


def vector_variance(rep: torch.Tensor) -> torch.Tensor:
    """Variance (unbiased) over the node dimension."""
    return rep.var(dim=0, unbiased=True)


def mean_activation_per_layer(reps):
    """
    Compute the global mean activation per list of representations.
    """
    return [rep.mean().item() for rep in reps]


def mean_node_activation_per_layer(reps):
    """
    Compute per-layer mean activation averaged over nodes (vector per layer).
    """
    return [vector_mean(rep).cpu() for rep in reps]


# Model persistence utilities
def save_checkpoint(
    sol_dir: Path,
    layer_str: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    best_val: float,
    layer_sizes: Sequence[int],
    dropout: float,
) -> None:
    """
    Save training state to disk.

    Args:
        sol_dir: Directory to save checkpoints into.
        layer_str: A string identifier for the architecture (e.g. "64-32-16").
        epoch: Current epoch number.
        model: The model being trained.
        optimizer: The optimizer.
        scheduler: Learning-rate scheduler.
        best_val: Best validation accuracy so far.
        layer_sizes: Sizes of each hidden layer.
        dropout: Dropout probability.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val,
        'layer_sizes': layer_sizes,
        'dropout': dropout,
    }
    ckpt_path = sol_dir / f"best_val_{layer_str}_ep{epoch}.pt"
    torch.save(checkpoint, ckpt_path)
    logging.info(f"Saved checkpoint: {ckpt_path}")


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
