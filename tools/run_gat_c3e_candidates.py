from __future__ import annotations

import argparse
import ast
import csv
import random
import sys
from dataclasses import dataclass
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

from Model_factory.activations import (  # noqa: E402
    ACTIVATION_KINDS,
    ACTIVATION_MODE_CHOICES,
    build_activation,
    normalize_activation_mode,
)
from Implementations.datasets import (  # noqa: E402
    DATASET_CHOICES,
    dataset_slug,
    load_node_dataset_with_splits,
)


@dataclass
class Candidate:
    candidate_id: int
    depth: int
    original_widths: list[int]
    snapped_widths: list[int]
    dropouts: list[float]
    phi0: float
    channel_capacity: float


class DeepResidualGAT(nn.Module):
    """Residual-connected multi-layer GAT using total-width layer sizes."""

    def __init__(
        self,
        *,
        in_channels: int,
        layer_widths: list[int],
        num_classes: int,
        dropouts: list[float],
        heads: int,
        attention_dropout: float,
        activation_mode: str = "first-on",
        activation_kind: str = "prelu",
    ) -> None:
        super().__init__()
        if not layer_widths:
            raise ValueError("layer_widths must not be empty.")
        if len(dropouts) != len(layer_widths):
            raise ValueError(
                f"dropouts length must be {len(layer_widths)}, got {len(dropouts)}"
            )
        if any(width % heads != 0 for width in layer_widths):
            raise ValueError("All layer widths must be divisible by heads.")

        self.heads = int(heads)
        self.activation_mode = normalize_activation_mode(activation_mode)
        self.activation_kind = str(activation_kind).strip().lower()
        self.layer_widths = [int(width) for width in layer_widths]
        self.channels_per_head = [width // heads for width in self.layer_widths]
        self.propagation = nn.ModuleList()
        previous_width = int(in_channels)
        for width in self.layer_widths:
            self.propagation.append(
                GATConv(
                    previous_width,
                    width // heads,
                    heads=heads,
                    concat=True,
                    dropout=attention_dropout,
                    add_self_loops=False,
                )
            )
            previous_width = width

        self.activations = nn.ModuleList(
            [build_activation(self.activation_kind, width) for width in self.layer_widths]
        )
        self.residuals = nn.ModuleList(
            [
                nn.Linear(self.layer_widths[idx - 1], self.layer_widths[idx])
                for idx in range(1, len(self.layer_widths))
            ]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(p=float(prob)) for prob in dropouts])
        self.classifier = nn.Linear(self.layer_widths[-1], num_classes)

    def uses_activation(self, layer_idx: int) -> bool:
        if self.activation_mode == "all-on":
            return True
        if self.activation_mode == "first-on":
            return layer_idx == 0
        return False

    def forward(self, data):
        h = data.x
        for idx, conv in enumerate(self.propagation):
            h_new = conv(h, data.edge_index)
            if idx > 0:
                h_new = h_new + self.residuals[idx - 1](h)
            if self.uses_activation(idx):
                h_new = self.activations[idx](h_new)
            h = self.dropouts[idx](h_new)
        return self.classifier(h)


def snap_width_down(width: int, heads: int) -> int:
    """Round ``width`` down to a positive multiple of ``heads``."""

    if width <= 0:
        raise ValueError("width must be positive.")
    if heads <= 0:
        raise ValueError("heads must be positive.")
    snapped = (int(width) // int(heads)) * int(heads)
    if snapped <= 0:
        raise ValueError(f"width {width} is too small for heads={heads}.")
    return snapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C3E-estimated residual GAT candidates.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="Cora")
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=None,
        help="C3E candidate summary CSV. Defaults to the newest dataset-matching GAT C3E summary.",
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--attention-dropout", type=float, default=0.6)
    parser.add_argument("--activation-mode", choices=ACTIVATION_MODE_CHOICES, default="first-on")
    parser.add_argument("--activation-kind", choices=ACTIVATION_KINDS, default="prelu")
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
    if args.candidate_summary is None:
        try:
            args.candidate_summary = latest_candidate_summary(args.results_root, args.dataset)
        except FileNotFoundError as exc:
            parser.error(str(exc))
    if not args.candidate_summary.exists():
        parser.error(f"--candidate-summary does not exist: {args.candidate_summary}")
    args.activation_mode = normalize_activation_mode(args.activation_mode)
    args.activation_kind = args.activation_kind.strip().lower()
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
    return f"{dataset_slug(args.dataset)}_gat_c3e_train_{stamp}"


def latest_candidate_summary(results_root: Path, dataset: str = "Cora") -> Path:
    prefix = dataset_slug(dataset)
    candidates = sorted(
        results_root.glob(f"{prefix}_gat_c3e_*/summary.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No GAT C3E candidate summary found. "
            "Run tools/inspect_gat_c3e.py first or pass --candidate-summary."
        )
    return candidates[0]


def load_candidates(path: Path, heads: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle), start=1):
            original_widths = [int(width) for width in ast.literal_eval(row["widths"])]
            dropouts = [float(prob) for prob in ast.literal_eval(row["dropouts"])]
            snapped_widths = [snap_width_down(width, heads) for width in original_widths]
            candidates.append(
                Candidate(
                    candidate_id=idx,
                    depth=int(row["depth"]),
                    original_widths=original_widths,
                    snapped_widths=snapped_widths,
                    dropouts=dropouts,
                    phi0=float(row["phi0"]),
                    channel_capacity=float(row["channel_capacity"]),
                )
            )
    if not candidates:
        raise ValueError(f"No candidates found in {path}.")
    return candidates


def load_data(args: argparse.Namespace):
    return load_node_dataset_with_splits(
        args.dataset,
        args.data_root,
        train_per_class=args.train_per_class,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
    )


load_cora = load_data


def evaluate(model: nn.Module, data) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data)
        labels = data.y.view(-1)
        pred = logits.argmax(dim=1)
        val_acc = float((pred[data.val_mask] == labels[data.val_mask]).sum().item()) / float(
            data.val_mask.sum().item()
        )
        test_acc = float((pred[data.test_mask] == labels[data.test_mask]).sum().item()) / float(
            data.test_mask.sum().item()
        )
    return val_acc, test_acc


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def train_candidate(
    *,
    candidate: Candidate,
    args: argparse.Namespace,
    data,
    dataset,
    device: str,
    save_root: Path,
) -> dict[str, Any]:
    set_seed(args.seed)
    data = data.clone().to(device)
    activation_mode = normalize_activation_mode(args.activation_mode)
    activation_kind = args.activation_kind.strip().lower()
    model = DeepResidualGAT(
        in_channels=data.x.size(1),
        layer_widths=candidate.snapped_widths,
        num_classes=dataset.num_classes,
        dropouts=candidate.dropouts,
        heads=args.heads,
        attention_dropout=args.attention_dropout,
        activation_mode=activation_mode,
        activation_kind=activation_kind,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    candidate_dir = save_root / f"candidate_depth_{candidate.depth}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = candidate_dir / "best.pt"

    best_val = float("-inf")
    best_test = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        labels = data.y.view(-1)
        loss = F.cross_entropy(logits[data.train_mask], labels[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc, test_acc = evaluate(model, data)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "candidate_id": candidate.candidate_id,
                    "depth": candidate.depth,
                    "original_widths": candidate.original_widths,
                    "snapped_widths": candidate.snapped_widths,
                    "dropouts": candidate.dropouts,
                    "heads": args.heads,
                    "activation_mode": activation_mode,
                    "activation_kind": activation_kind,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "test_acc_at_best_val": best_test,
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            break

    return {
        "candidate_id": candidate.candidate_id,
        "depth": candidate.depth,
        "original_widths": candidate.original_widths,
        "snapped_widths": candidate.snapped_widths,
        "heads": args.heads,
        "activation_mode": activation_mode,
        "activation_kind": activation_kind,
        "channels_per_head": [width // args.heads for width in candidate.snapped_widths],
        "dropouts": candidate.dropouts,
        "phi0": candidate.phi0,
        "channel_capacity": candidate.channel_capacity,
        "status": "success",
        "device": device,
        "epochs_run": final_epoch,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "test_at_best_val": best_test,
        "checkpoint": str(checkpoint_path),
        "error": "",
    }


def failed_row(candidate: Candidate, device: str, error: str, args: argparse.Namespace) -> dict[str, Any]:
    activation_mode = normalize_activation_mode(args.activation_mode)
    activation_kind = args.activation_kind.strip().lower()
    return {
        "candidate_id": candidate.candidate_id,
        "depth": candidate.depth,
        "original_widths": candidate.original_widths,
        "snapped_widths": candidate.snapped_widths,
        "heads": args.heads,
        "activation_mode": activation_mode,
        "activation_kind": activation_kind,
        "channels_per_head": [width // args.heads for width in candidate.snapped_widths],
        "dropouts": candidate.dropouts,
        "phi0": candidate.phi0,
        "channel_capacity": candidate.channel_capacity,
        "status": "failed",
        "device": device,
        "epochs_run": "",
        "best_epoch": "",
        "best_val": "",
        "test_at_best_val": "",
        "checkpoint": "",
        "error": error,
    }


def row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (repr(value) if isinstance(value, list) else value)
        for key, value in row.items()
    }


def write_outputs(result_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "depth",
        "original_widths",
        "snapped_widths",
        "heads",
        "activation_mode",
        "activation_kind",
        "channels_per_head",
        "dropouts",
        "phi0",
        "channel_capacity",
        "status",
        "device",
        "epochs_run",
        "best_epoch",
        "best_val",
        "test_at_best_val",
        "checkpoint",
        "error",
    ]
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = row_for_csv(row)
            writer.writerow({field: csv_row.get(field, "") for field in fields})

    successful = [row for row in rows if row.get("status") == "success"]
    ranking = sorted(
        successful,
        key=lambda row: (float(row["test_at_best_val"]), float(row["best_val"])),
        reverse=True,
    )

    lines = [
        f"# {args.dataset} GAT C3E Candidate Training",
        "",
        "## Settings",
        "",
        f"- Dataset: {args.dataset}",
        f"- Candidate summary: `{args.candidate_summary}`",
        f"- Heads: {args.heads}",
        f"- Activation mode: {args.activation_mode}",
        f"- Activation kind: {args.activation_kind}",
        "- Width snapping: round down to nearest positive multiple of heads",
        f"- Epochs/patience: {args.epochs}/{args.patience}",
        f"- LR/weight_decay: {args.lr}/{args.weight_decay}",
        f"- Attention dropout: {args.attention_dropout}",
        f"- Seed: {args.seed}",
        "",
        "## Ranking",
        "",
        "| Rank | Depth | Test@best val | Best val | Best epoch | Device | Snapped widths |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | {row['depth']} | {float(row['test_at_best_val']):.4f} | "
            f"{float(row['best_val']):.4f} | {row['best_epoch']} | {row['device']} | "
            f"`{row['snapped_widths']}` |"
        )

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Depth | Original widths | Snapped widths | Val | Test | Epochs | Checkpoint |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if row["status"] == "success":
            lines.append(
                f"| {row['depth']} | `{row['original_widths']}` | `{row['snapped_widths']}` | "
                f"{float(row['best_val']):.4f} | {float(row['test_at_best_val']):.4f} | "
                f"{row['epochs_run']} | `{row['checkpoint']}` |"
            )
        else:
            lines.append(
                f"| {row['depth']} | `{row['original_widths']}` | `{row['snapped_widths']}` | "
                f" |  |  | failed: {row['error']} |"
            )

    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    name = run_name(args)
    save_root = args.saved_root / name
    result_dir = args.results_root / name
    save_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.candidate_summary, args.heads)
    dataset, data = load_data(args)
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        try:
            print(
                f"Training depth={candidate.depth} widths={candidate.snapped_widths} on {device}...",
                flush=True,
            )
            row = train_candidate(
                candidate=candidate,
                args=args,
                data=data,
                dataset=dataset,
                device=device,
                save_root=save_root,
            )
            rows.append(row)
        except RuntimeError as exc:
            if device == "cuda" and is_cuda_oom(exc):
                torch.cuda.empty_cache()
                print(f"CUDA OOM at depth={candidate.depth}; retrying on CPU...", flush=True)
                try:
                    row = train_candidate(
                        candidate=candidate,
                        args=args,
                        data=data,
                        dataset=dataset,
                        device="cpu",
                        save_root=save_root,
                    )
                    rows.append(row)
                except Exception as cpu_exc:
                    rows.append(failed_row(candidate, "cpu", str(cpu_exc), args))
            else:
                rows.append(failed_row(candidate, device, str(exc), args))
        except Exception as exc:
            rows.append(failed_row(candidate, device, str(exc), args))
        write_outputs(result_dir, rows, args)

    print(f"Results written to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
