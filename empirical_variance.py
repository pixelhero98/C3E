"""Empirical propagation-variance utilities for learned message passing layers."""

from __future__ import annotations

import math
from typing import Any

import torch


EPS = 1e-12


def attention_variance_stats(edge_index: torch.Tensor, alpha: torch.Tensor, num_nodes: int) -> dict[str, Any]:
    """
    Compute sparse attention-matrix variance over all ``num_nodes ** 2`` entries.

    ``alpha`` may be shaped ``[num_edges]`` or ``[num_edges, heads]``. Duplicate edges
    are coalesced before computing per-head statistics.
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if not torch.is_floating_point(alpha):
        alpha = alpha.float()
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)
    if alpha.dim() != 2:
        raise ValueError("alpha must have shape [num_edges] or [num_edges, heads].")
    if edge_index.size(1) != alpha.size(0):
        raise ValueError("edge_index and alpha must describe the same number of edges.")
    if not torch.isfinite(alpha).all():
        raise ValueError("attention weights must be finite.")

    edge_index = edge_index.detach().cpu().long()
    alpha = alpha.detach().cpu().double()
    total_entries = float(num_nodes * num_nodes)
    per_head: list[float] = []

    for head_idx in range(alpha.size(1)):
        sparse = torch.sparse_coo_tensor(
            edge_index,
            alpha[:, head_idx],
            size=(num_nodes, num_nodes),
            dtype=torch.double,
        ).coalesce()
        values = sparse.values()
        total = float(values.sum().item())
        sum_sq = float((values * values).sum().item())
        mean_value = total / total_entries
        mean_square = sum_sq / total_entries
        per_head.append(max(mean_square - mean_value * mean_value, 0.0))

    sorted_variances = sorted(per_head)
    head_count = len(per_head)
    if head_count % 2:
        median = sorted_variances[head_count // 2]
    else:
        median = 0.5 * (
            sorted_variances[head_count // 2 - 1] + sorted_variances[head_count // 2]
        )
    geometric_mean = math.exp(sum(math.log(value + EPS) for value in per_head) / head_count)

    return {
        "heads": head_count,
        "per_head": per_head,
        "mean": sum(per_head) / head_count,
        "median": median,
        "max": max(per_head),
        "geometric_mean": geometric_mean,
    }
