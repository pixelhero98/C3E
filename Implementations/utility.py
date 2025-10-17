import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def create_masks(
    data: Data,
    train_per_class: int,
    num_val: int,
    num_test: int,
    *,
    labels: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate boolean train/val/test masks with a fixed number of training
    examples per class, then random validation & test splits from the remainder.

    Args:
        data (Data): PyG Data object with node features & labels.
        train_per_class (int): Number of train nodes to sample from each class.
        num_val (int): Total number of validation nodes.
        num_test (int): Total number of test nodes.
        labels (Tensor, optional): 1D long tensor of node labels.  If None,
            will use `data.y`.
        num_classes (int, optional): Number of classes.  If None, inferred as
            `labels.max().item() + 1`.
        seed (int, optional): If set, seeds the RNG for fixed splits.

    Returns:
        train_mask, val_mask, test_mask (Tensor[bool], each of shape [num_nodes])
    """
    if train_per_class <= 0 or num_val < 0 or num_test < 0:
        raise ValueError("train_per_class must be > 0 and num_val/num_test must be >= 0")

    # 1) get labels & num_classes
    y = labels if labels is not None else data.y
    y = y.view(-1)
    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    if seed is not None:
        device = y.device if y.is_cuda else torch.device('cpu')
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None

    N = y.size(0)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    # 2) sample train_per_class from each class
    for c in range(num_classes):
        idx_c = (y == c).nonzero(as_tuple=False).view(-1)
        if idx_c.numel() < train_per_class:
            raise ValueError(
                f"Class {c} has only {idx_c.numel()} samples, "
                f"but you requested {train_per_class}"
            )
        perm = idx_c[torch.randperm(idx_c.size(0), generator=generator, device=idx_c.device)]
        train_mask[perm[:train_per_class]] = True

    # 3) split the rest into val & test
    rem = (~train_mask).nonzero(as_tuple=False).view(-1)
    rem = rem[torch.randperm(rem.size(0), generator=generator, device=rem.device)]
    if rem.numel() < (num_val + num_test):
        raise ValueError(
            f"Not enough remaining samples ({rem.numel()}) for "
            f"{num_val} val + {num_test} test"
        )
    val_mask[rem[:num_val]] = True
    test_mask[rem[num_val : num_val + num_test]] = True

    return train_mask, val_mask, test_mask
    
# Ensure your model and data live on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model: nn.Module, data: Data, optimizer: optim.Optimizer) -> float:
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
    if data.train_mask.sum() == 0:
        raise ValueError("Training mask is empty; cannot compute loss.")
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def val(model: nn.Module, data: Data) -> float:
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
    if total == 0:
        raise ValueError("Validation mask is empty; cannot compute accuracy.")
    return correct / total

def test(model: nn.Module, data: Data) -> float:
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
    if total == 0:
        raise ValueError("Test mask is empty; cannot compute accuracy.")
    return correct / total
    
def get_model_rep(model: nn.Module, data: Data):
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
    if not hasattr(model, "propagation"):
        raise AttributeError("Model does not define a 'propagation' attribute for convolution layers.")
    if not hasattr(model, "activations"):
        raise AttributeError("Model does not define an 'activations' attribute for activation layers.")

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

def rep_entropy(rep: Tensor, nbins: int = 5000) -> float:
    """
    Estimate Shannon entropy of a flattened tensor by:
      1. Softmax to map values into [0,1]
      2. Histogram into `nbins` bins over [0,1]
      3. Compute entropy = -sum(p_i log p_i)
    """
    rep_flat = rep.view(-1)
    p = F.softmax(rep_flat, dim=0)
    p_cpu = p.detach().cpu()

    counts = torch.histc(p_cpu, bins=nbins, min=0.0, max=1.0)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total

    eps = 1e-12
    entropy = -(probs * torch.log(probs + eps)).sum().item()
    return float(entropy)

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

    return entropies

def compute_sparse_laplacian(edge_index: Tensor, num_nodes: int) -> SparseTensor:
    row, col = edge_index
    values = torch.ones(row.size(0), device=row.device, dtype=torch.float32)
    adjacency = SparseTensor(row=row, col=col, value=values, sparse_sizes=(num_nodes, num_nodes))
    deg = adjacency.sum(dim=1)
    degree_matrix = SparseTensor.diag(deg)
    return degree_matrix - adjacency

def compute_normalized_laplacian(edge_index: Tensor, num_nodes: int) -> SparseTensor:
    row, col = edge_index
    values = torch.ones(row.size(0), device=row.device, dtype=torch.float32)
    adjacency = SparseTensor(row=row, col=col, value=values, sparse_sizes=(num_nodes, num_nodes))
    deg = adjacency.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(~torch.isfinite(deg_inv_sqrt), 0)
    deg_inv_sqrt = deg_inv_sqrt.clamp_min(0)
    d_inv_sqrt = SparseTensor.diag(deg_inv_sqrt)
    identity = SparseTensor.eye(num_nodes, device=deg.device)
    return identity - d_inv_sqrt.matmul(adjacency).matmul(d_inv_sqrt)

# New wrapper to switch between Laplacians:
def compute_laplacian(
    edge_index,
    num_nodes: int,
    normalized: bool = False
) -> SparseTensor:
    """
    Build either the unnormalized (D - A) or symmetric normalized (I - D^{-1/2} A D^{-1/2}) Laplacian.

    Args:
        edge_index: edge index tensor of shape [2, E]
        num_nodes: number of nodes in the graph
        normalized: if True, returns normalized Laplacian; else unnormalized.

    Returns:
        SparseTensor: Laplacian matrix
    """
    if normalized:
        return compute_normalized_laplacian(edge_index, num_nodes)
    else:
        return compute_sparse_laplacian(edge_index, num_nodes)

# Updated energy display to accept a choice:
def compute_dirichlet_energy(rep: Tensor, laplacian: SparseTensor) -> float:
    r"""Compute Dirichlet energy :math:`\sum_i f_i (Lf)_i` for a representation."""
    if rep.dim() == 1:
        rep = rep.unsqueeze(-1)

    laplacian = laplacian.to(rep.device)
    dtype_attr = getattr(laplacian, "dtype", None)
    if callable(dtype_attr):
        dtype = dtype_attr()
    elif dtype_attr is not None:
        dtype = dtype_attr
    else:
        dtype = rep.dtype
    rep = rep.to(dtype)
    rep_laplacian = laplacian.matmul(rep)
    energy = (rep * rep_laplacian).sum().item()
    return float(energy)


def show_layer_dirichlet_energy(
    conv_reps, act_reps, data: Data,
    normalized: bool = False
):
    """
    Print normalized Dirichlet energy (relative to input) for input, conv, and activation layers.

    Args:
        conv_reps: list of conv-layer outputs (cpu, detached)
        act_reps: list of activation-layer outputs (cpu, detached)
        data: PyG data object with x and edge_index
        normalized: whether to use the symmetric normalized Laplacian
    """
    cpu_data = data.to('cpu')
    laplacian = compute_laplacian(cpu_data.edge_index, cpu_data.x.size(0), normalized=normalized)

    base_energy = compute_dirichlet_energy(cpu_data.x, laplacian)
    if base_energy == 0:
        logging.warning("Input representation has zero Dirichlet energy; using 1.0 to avoid division by zero.")
        base_energy = 1.0

    energies = {'input': 1.0}

    for idx, rep in enumerate(conv_reps, start=1):
        energies[f'conv_{idx}'] = compute_dirichlet_energy(rep, laplacian) / base_energy
    for idx, rep in enumerate(act_reps, start=1):
        energies[f'act_{idx}'] = compute_dirichlet_energy(rep, laplacian) / base_energy

    print(energies)
    print('===============================================================')

    return energies

def vector_mean(rep: torch.Tensor) -> torch.Tensor:
    """Mean over the node dimension."""
    return rep.mean(dim=0)

def vector_variance(rep: torch.Tensor) -> torch.Tensor:
    """Variance (unbiased) over the node dimension."""
    if rep.size(0) <= 1:
        shape = rep.shape[1:]
        return torch.zeros(shape, dtype=rep.dtype, device=rep.device)
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
    dropout: Sequence[float],
) -> Path:
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
        dropout: Dropout probabilities for each layer.

    Returns:
        The path to the saved checkpoint.
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
    return ckpt_path

def load_model(
    model_class,
    path: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
    *model_args,
    **model_kwargs
) -> torch.nn.Module:
    """
    Instantiate a model for inference:

    1. Builds `model_class(*model_args, **model_kwargs)`.
    2. Loads weights from `path`.
    3. Moves to `device` and sets to eval() mode.

    Args:
        model_class: class or factory for your nn.Module.
        path: Path to a checkpoint file saved via `save_checkpoint`.
        device: device string or torch.device (default 'cpu').
        *model_args, **model_kwargs: parameters for model_class constructor.

    Returns:
        An nn.Module on `device`, ready for inference.

    Example usage:
        # Assume you trained and saved a checkpoint under 'checkpoints/best_model.pt'
        from MyModelModule import MyGNNModel

        model = load_model(
            Model,
            path='checkpoints/best_model.pt',
            device='cuda',
            prop_layer=[input_dim, 64, 32, output_dim],
            num_class=dataset.num_classes,
            drop_probs=[0.5, 0.5, 0.0],
            use_activations=[True, True, False],
            conv_methods=['gcn', 'gcn', 'gcn']
        )
        # Now `model` is on GPU and in eval() mode, ready for inference.
    """
    device = torch.device(device)
    model = model_class(*model_args, **model_kwargs)
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
