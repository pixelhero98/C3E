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
    labels: Optional[Tensor] = None,
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Create boolean train/val/test masks from node labels, sampling
    `train_per_class` nodes per class for training, then `num_val` nodes
    for validation and `num_test` nodes for testing from the remaining.

    Args:
        data: PyG Data object with `y`.
        train_per_class: number of nodes per class for training.
        num_val: number of nodes for validation.
        num_test: number of nodes for testing.
        labels: optional labels tensor to use instead of `data.y`.
        num_classes: optional number of classes.
        seed: optional random seed.

    Returns:
        (train_mask, val_mask, test_mask)
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

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=y.device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=y.device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=y.device)

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
    y = data.y.view(-1)
    loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
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
        y = data.y.view(-1)
        correct = int((pred[data.val_mask] == y[data.val_mask]).sum())
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
        y = data.y.view(-1)
        correct = int((pred[data.test_mask] == y[data.test_mask]).sum())
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
    device = next(model.parameters()).device
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
    entropies = {'input': rep_entropy(data.x, nbins)}

    for idx, rep in enumerate(conv_reps, start=1):
        entropies[f'conv_{idx}'] = rep_entropy(rep, nbins)
    for idx, rep in enumerate(act_reps, start=1):
        entropies[f'act_{idx}'] = rep_entropy(rep, nbins)

    print(entropies)
    print('===============================================================')

    return entropies


def show_layer_dirichlet_energy(conv_reps, act_reps, data, normalized: bool = True):
    """
    Compute Dirichlet energy for input features and each conv and activation layer.
    """
    laplacian = compute_laplacian(data.edge_index, data.num_nodes, normalized=normalized)
    energies = {"input": compute_dirichlet_energy(data.x, laplacian)}

    for idx, rep in enumerate(conv_reps, start=1):
        energies[f"conv_{idx}"] = compute_dirichlet_energy(rep, laplacian)
    for idx, rep in enumerate(act_reps, start=1):
        energies[f"act_{idx}"] = compute_dirichlet_energy(rep, laplacian)

    print(energies)
    print("===============================================================")

    return energies


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
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt_matrix = SparseTensor.diag(deg_inv_sqrt)
    identity = SparseTensor.eye(num_nodes, device=deg.device)
    normalized_adj = deg_inv_sqrt_matrix @ adjacency @ deg_inv_sqrt_matrix
    return identity - normalized_adj


def compute_laplacian(edge_index: Tensor, num_nodes: int, normalized: bool = True) -> SparseTensor:
    """
    Compute the graph Laplacian.
    """
    if normalized:
        return compute_normalized_laplacian(edge_index, num_nodes)
    else:
        return compute_sparse_laplacian(edge_index, num_nodes)


def compute_dirichlet_energy(rep: Tensor, laplacian: SparseTensor) -> float:
    r"""Compute Dirichlet energy :math:`\sum_i f_i (Lf)_i` for a representation."""
    if rep.dim() == 1:
        rep = rep.unsqueeze(-1)

    laplacian = laplacian.to(rep.device)
    rep = rep.to(rep.device)

    # Lf
    Lf = laplacian.matmul(rep)
    energy = (rep * Lf).sum().item()
    return float(energy)


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


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    device: Union[str, torch.device] = 'cpu',
) -> dict:
    """
    Load checkpoint into model and optionally optimizer/scheduler.
    """
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def load_model(
    model_class,
    path: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
    *model_args,
    **model_kwargs,
) -> nn.Module:
    """
    Load a model from a checkpoint file.

    Args:
        model_class: the nn.Module class to instantiate.
        path: path to checkpoint.
        device: cpu/cuda.
        model_args/model_kwargs: constructor args.

    Returns:
        An nn.Module on `device`, ready for inference.
    """
    model = model_class(*model_args, **model_kwargs)
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
