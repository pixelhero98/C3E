from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Planetoid, WikipediaNetwork

from Implementations.utility import create_masks


DATASET_CHOICES = (
    "Cora",
    "CiteSeer",
    "PubMed",
    "Chameleon",
    "Squirrel",
    "AmazonPhoto",
    "AmazonComputers",
    "ogbn-arxiv",
    "ogbn-papers100M",
)

OGB_DATASETS = {"ogbn-arxiv", "ogbn-papers100M"}


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace("-", "_")


def _load_ogb_dataset_class():
    try:
        module = import_module("ogb.nodeproppred")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ogb is required for ogbn-* datasets. Install ogb or choose a non-OGB dataset."
        ) from exc
    return module.PygNodePropPredDataset


def load_node_dataset(dataset_name: str, data_root: Path):
    if dataset_name in {"Cora", "CiteSeer", "PubMed"}:
        return Planetoid(str(data_root), name=dataset_name, transform=T.AddSelfLoops())
    if dataset_name in {"AmazonPhoto", "AmazonComputers"}:
        amazon_name = "Photo" if dataset_name == "AmazonPhoto" else "Computers"
        return Amazon(str(data_root), name=amazon_name, transform=T.AddSelfLoops())
    if dataset_name in {"Chameleon", "Squirrel"}:
        return WikipediaNetwork(str(data_root), name=dataset_name.lower(), transform=T.AddSelfLoops())
    if dataset_name in OGB_DATASETS:
        dataset_class = _load_ogb_dataset_class()
        return dataset_class(root=str(data_root), name=dataset_name, transform=T.AddSelfLoops())
    raise ValueError(f"Unsupported dataset {dataset_name!r}. Expected one of {DATASET_CHOICES}.")


def apply_node_splits(
    dataset: Any,
    data,
    dataset_name: str,
    *,
    train_per_class: int,
    num_val: int,
    num_test: int,
    seed: int,
) -> None:
    if dataset_name in OGB_DATASETS:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"]
        test_idx = split_idx["test"]

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[valid_idx] = True
        data.test_mask[test_idx] = True
        return

    data.train_mask, data.val_mask, data.test_mask = create_masks(
        data,
        train_per_class,
        num_val,
        num_test,
        seed=seed,
    )


def load_node_dataset_with_splits(
    dataset_name: str,
    data_root: Path,
    *,
    train_per_class: int = 20,
    num_val: int = 500,
    num_test: int = 1000,
    seed: int = 42,
):
    dataset = load_node_dataset(dataset_name, data_root)
    data = dataset[0]
    apply_node_splits(
        dataset,
        data,
        dataset_name,
        train_per_class=train_per_class,
        num_val=num_val,
        num_test=num_test,
        seed=seed,
    )
    return dataset, data
