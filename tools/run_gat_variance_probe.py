from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Implementations.empirical_variance import attention_variance_stats  # noqa: E402
from Implementations.datasets import (  # noqa: E402
    DATASET_CHOICES,
    dataset_slug,
    load_node_dataset_with_splits,
)


class OneLayerGAT(nn.Module):
    """One hidden GAT layer followed by a linear classifier."""

    def __init__(
        self,
        in_channels: int,
        hidden_width: int,
        num_classes: int,
        *,
        heads: int,
        attention_dropout: float,
        feature_dropout: float,
    ) -> None:
        super().__init__()
        if hidden_width % heads != 0:
            raise ValueError("hidden_width must be divisible by heads.")
        self.hidden_width = int(hidden_width)
        self.heads = int(heads)
        self.channels_per_head = hidden_width // heads
        self.feature_dropout = float(feature_dropout)
        self.gat = GATConv(
            in_channels,
            self.channels_per_head,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            add_self_loops=False,
        )
        self.classifier = nn.Linear(hidden_width, num_classes)

    def forward(self, data):
        h = F.dropout(data.x, p=self.feature_dropout, training=self.training)
        h = self.gat(h, data.edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.feature_dropout, training=self.training)
        return self.classifier(h)

    def attention_weights(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        _, attention = self.gat(data.x, data.edge_index, return_attention_weights=True)
        edge_index, alpha = attention
        return edge_index, alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-layer GATs and estimate attention variance.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="Cora")
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--attention-dropout", type=float, default=0.6)
    parser.add_argument("--feature-dropout", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--num-val", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--saved-root", type=Path, default=REPO_ROOT / "saved")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if args.heads <= 0:
        parser.error("--heads must be positive.")
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("--epochs and --patience must be positive.")
    for width in args.widths:
        if width <= 0:
            parser.error("--widths must be positive.")
        if width % args.heads != 0:
            parser.error(f"width {width} is not divisible by heads={args.heads}.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_slug(args.dataset)}_gat_variance_probe_{stamp}"


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def evaluate(model: nn.Module, data) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=1)
        labels = data.y.view(-1)
        val_acc = float((pred[data.val_mask] == labels[data.val_mask]).sum().item()) / float(
            data.val_mask.sum().item()
        )
        test_acc = float((pred[data.test_mask] == labels[data.test_mask]).sum().item()) / float(
            data.test_mask.sum().item()
        )
    return val_acc, test_acc


def snapshot_row(
    *,
    model: OneLayerGAT,
    data,
    width: int,
    snapshot: str,
    epoch: int,
    device: str,
    val_acc: float,
    test_acc: float,
    best_checkpoint: Path,
    status: str = "success",
    error: str = "",
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        edge_index, alpha = model.attention_weights(data)
    stats = attention_variance_stats(edge_index, alpha, data.num_nodes)
    return {
        "width": width,
        "heads": model.heads,
        "channels_per_head": model.channels_per_head,
        "snapshot": snapshot,
        "epoch": epoch,
        "device": device,
        "status": status,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "attention_edges": int(edge_index.size(1)),
        "attention_heads": int(stats["heads"]),
        "sigma_empirical": float(stats["geometric_mean"]),
        "variance_mean": float(stats["mean"]),
        "variance_median": float(stats["median"]),
        "variance_max": float(stats["max"]),
        "variance_per_head": json.dumps(stats["per_head"]),
        "checkpoint": str(best_checkpoint),
        "error": error,
    }


def failed_row(width: int, device: str, error: str) -> dict[str, Any]:
    return {
        "width": width,
        "heads": "",
        "channels_per_head": "",
        "snapshot": "failed",
        "epoch": "",
        "device": device,
        "status": "failed",
        "val_acc": "",
        "test_acc": "",
        "attention_edges": "",
        "attention_heads": "",
        "sigma_empirical": "",
        "variance_mean": "",
        "variance_median": "",
        "variance_max": "",
        "variance_per_head": "",
        "checkpoint": "",
        "error": error,
    }


def train_one_width(
    *,
    args: argparse.Namespace,
    data,
    dataset,
    width: int,
    device: str,
    save_root: Path,
) -> list[dict[str, Any]]:
    set_seed(args.seed)
    data = data.clone().to(device)
    model = OneLayerGAT(
        in_channels=data.x.size(1),
        hidden_width=width,
        num_classes=dataset.num_classes,
        heads=args.heads,
        attention_dropout=args.attention_dropout,
        feature_dropout=args.feature_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    width_dir = save_root / f"width_{width}"
    width_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = width_dir / "best.pt"

    rows: list[dict[str, Any]] = []
    init_val, init_test = evaluate(model, data)
    rows.append(
        snapshot_row(
            model=model,
            data=data,
            width=width,
            snapshot="init",
            epoch=0,
            device=device,
            val_acc=init_val,
            test_acc=init_test,
            best_checkpoint=best_checkpoint,
        )
    )

    best_val = float("-inf")
    best_test = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    epoch1_row: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.y.view(-1)[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc, test_acc = evaluate(model, data)
        if epoch == 1:
            epoch1_row = snapshot_row(
                model=model,
                data=data,
                width=width,
                snapshot="epoch1",
                epoch=epoch,
                device=device,
                val_acc=val_acc,
                test_acc=test_acc,
                best_checkpoint=best_checkpoint,
            )

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "width": width,
                    "heads": args.heads,
                    "channels_per_head": width // args.heads,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "test_acc_at_best_val": best_test,
                },
                best_checkpoint,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            break

    if epoch1_row is None:
        raise RuntimeError("Training did not complete epoch 1.")
    rows.append(epoch1_row)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_val, best_test = evaluate(model, data)
    rows.append(
        snapshot_row(
            model=model,
            data=data,
            width=width,
            snapshot="best",
            epoch=best_epoch,
            device=device,
            val_acc=best_val,
            test_acc=best_test,
            best_checkpoint=best_checkpoint,
        )
    )
    return rows


def write_outputs(
    result_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    save_root: Path,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "width",
        "heads",
        "channels_per_head",
        "snapshot",
        "epoch",
        "device",
        "status",
        "val_acc",
        "test_acc",
        "attention_edges",
        "attention_heads",
        "sigma_empirical",
        "variance_mean",
        "variance_median",
        "variance_max",
        "variance_per_head",
        "checkpoint",
        "error",
    ]
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    successful = [row for row in rows if row.get("status") == "success"]
    best_rows = [row for row in successful if row["snapshot"] == "best"]
    sigmas = [float(row["sigma_empirical"]) for row in best_rows if row.get("sigma_empirical") != ""]
    stability_note = "No successful best-checkpoint rows were produced."
    if sigmas:
        ratio = max(sigmas) / max(min(sigmas), 1e-12)
        if ratio <= 2.0:
            label = "stable"
        elif ratio <= 5.0:
            label = "moderately width-dependent"
        else:
            label = "highly width-dependent"
        stability_note = (
            f"Best-checkpoint sigma_empirical spans {min(sigmas):.6e} to "
            f"{max(sigmas):.6e} ({ratio:.2f}x), which is {label} for calibration."
        )

    lines = [
        f"# {args.dataset} GAT Empirical Variance Probe",
        "",
        "## Settings",
        "",
        f"- Dataset: {args.dataset}",
        f"- Widths: {', '.join(str(width) for width in args.widths)}",
        f"- Heads: {args.heads}",
        "- Width semantics: total concatenated hidden width",
        f"- Epochs/patience: {args.epochs}/{args.patience}",
        f"- LR/weight_decay: {args.lr}/{args.weight_decay}",
        f"- Attention/feature dropout: {args.attention_dropout}/{args.feature_dropout}",
        f"- Seed: {args.seed}",
        f"- Save root: `{save_root}`",
        "",
        "## Calibration Note",
        "",
        stability_note,
        "",
        "## Snapshots",
        "",
        "| Width | Snapshot | Epoch | Device | Val | Test | Sigma empirical | Mean var | Median var | Max var |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in successful:
        lines.append(
            f"| {row['width']} | {row['snapshot']} | {row['epoch']} | {row['device']} | "
            f"{float(row['val_acc']):.4f} | {float(row['test_acc']):.4f} | "
            f"{float(row['sigma_empirical']):.6e} | {float(row['variance_mean']):.6e} | "
            f"{float(row['variance_median']):.6e} | {float(row['variance_max']):.6e} |"
        )

    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.append(f"- width {row['width']} on {row['device']}: {row['error']}")

    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_data(args: argparse.Namespace):
    return load_node_dataset_with_splits(
        args.dataset,
        args.data_root,
        train_per_class=args.train_per_class,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    name = run_name(args)
    save_root = args.saved_root / name
    result_dir = args.results_root / name
    save_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset, data = load_data(args)
    rows: list[dict[str, Any]] = []

    for width in args.widths:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        try:
            print(f"Running width={width} on {device}...", flush=True)
            rows.extend(
                train_one_width(
                    args=args,
                    data=data,
                    dataset=dataset,
                    width=width,
                    device=device,
                    save_root=save_root,
                )
            )
        except RuntimeError as exc:
            if device == "cuda" and is_cuda_oom(exc):
                torch.cuda.empty_cache()
                print(f"CUDA OOM for width={width}; retrying on CPU...", flush=True)
                try:
                    rows.extend(
                        train_one_width(
                            args=args,
                            data=data,
                            dataset=dataset,
                            width=width,
                            device="cpu",
                            save_root=save_root,
                        )
                    )
                except Exception as cpu_exc:
                    rows.append(failed_row(width, "cpu", str(cpu_exc)))
            else:
                rows.append(failed_row(width, device, str(exc)))
        except Exception as exc:
            rows.append(failed_row(width, device, str(exc)))
        write_outputs(result_dir, rows, args, save_root)

    print(f"Results written to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
