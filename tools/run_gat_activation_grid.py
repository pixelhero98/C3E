from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_factory.activations import (  # noqa: E402
    ACTIVATION_KINDS,
    ACTIVATION_MODES,
    ACTIVATION_MODE_CHOICES,
    normalize_activation_mode,
)
from Implementations.datasets import DATASET_CHOICES, dataset_slug  # noqa: E402
from tools.run_gat_c3e_candidates import (  # noqa: E402
    failed_row,
    is_cuda_oom,
    latest_candidate_summary,
    load_candidates,
    load_data,
    row_for_csv,
    train_candidate,
)


def activation_combinations(
    modes: list[str] | None = None,
    kinds: list[str] | None = None,
) -> list[tuple[str, str]]:
    canonical_modes: list[str] = []
    for mode in modes or list(ACTIVATION_MODES):
        canonical = normalize_activation_mode(mode)
        if canonical not in canonical_modes:
            canonical_modes.append(canonical)

    canonical_kinds: list[str] = []
    for kind in kinds or list(ACTIVATION_KINDS):
        key = str(kind).strip().lower()
        if key not in ACTIVATION_KINDS:
            raise ValueError(f"activation_kind must be one of {ACTIVATION_KINDS}, got {kind!r}.")
        if key not in canonical_kinds:
            canonical_kinds.append(key)

    return [(mode, kind) for mode in canonical_modes for kind in canonical_kinds]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GAT C3E activation placement/function grid on estimated candidates."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="Cora")
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=None,
        help="C3E candidate summary CSV. Defaults to the newest dataset-matching GAT C3E summary.",
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--attention-dropout", type=float, default=0.6)
    parser.add_argument(
        "--activation-modes",
        nargs="+",
        choices=ACTIVATION_MODE_CHOICES,
        default=list(ACTIVATION_MODES),
    )
    parser.add_argument(
        "--activation-kinds",
        nargs="+",
        choices=ACTIVATION_KINDS,
        default=list(ACTIVATION_KINDS),
    )
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
    args.activation_modes = [
        mode for mode, _ in activation_combinations(args.activation_modes, [ACTIVATION_KINDS[0]])
    ]
    args.activation_kinds = [
        kind for _, kind in activation_combinations([ACTIVATION_MODES[0]], args.activation_kinds)
    ]
    return args


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_slug(args.dataset)}_gat_c3e_activation_grid_h{args.heads}_{stamp}"


def combo_args(args: argparse.Namespace, activation_mode: str, activation_kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        heads=args.heads,
        attention_dropout=args.attention_dropout,
        activation_mode=activation_mode,
        activation_kind=activation_kind,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )


def write_outputs(
    result_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    save_root: Path,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "combo_id",
        "activation_mode",
        "activation_kind",
        "candidate_id",
        "depth",
        "original_widths",
        "snapped_widths",
        "heads",
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

    def best_by(group_key: str) -> list[dict[str, Any]]:
        winners: list[dict[str, Any]] = []
        for value in sorted({row[group_key] for row in successful}):
            group_rows = [row for row in successful if row[group_key] == value]
            winners.append(
                max(
                    group_rows,
                    key=lambda row: (float(row["test_at_best_val"]), float(row["best_val"])),
                )
            )
        return winners

    lines = [
        f"# {args.dataset} GAT C3E Activation Grid",
        "",
        "## Settings",
        "",
        f"- Dataset: {args.dataset}",
        f"- Candidate summary: `{args.candidate_summary}`",
        f"- Heads: {args.heads}",
        f"- Activation modes: {', '.join(args.activation_modes)}",
        f"- Activation kinds: {', '.join(args.activation_kinds)}",
        f"- Epochs/patience: {args.epochs}/{args.patience}",
        f"- LR/weight_decay: {args.lr}/{args.weight_decay}",
        f"- Attention dropout: {args.attention_dropout}",
        f"- Seed: {args.seed}",
        f"- Save root: `{save_root}`",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Mode | Kind | Depth | Test@best val | Best val | Best epoch | Device | Snapped widths |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | {row['activation_mode']} | {row['activation_kind']} | "
            f"{row['depth']} | {float(row['test_at_best_val']):.4f} | "
            f"{float(row['best_val']):.4f} | {row['best_epoch']} | {row['device']} | "
            f"`{row['snapped_widths']}` |"
        )

    lines.extend(
        [
            "",
            "## Best Per Activation Mode",
            "",
            "| Mode | Kind | Depth | Test@best val | Best val | Snapped widths |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in best_by("activation_mode"):
        lines.append(
            f"| {row['activation_mode']} | {row['activation_kind']} | {row['depth']} | "
            f"{float(row['test_at_best_val']):.4f} | {float(row['best_val']):.4f} | "
            f"`{row['snapped_widths']}` |"
        )

    lines.extend(
        [
            "",
            "## Best Per Activation Kind",
            "",
            "| Kind | Mode | Depth | Test@best val | Best val | Snapped widths |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in best_by("activation_kind"):
        lines.append(
            f"| {row['activation_kind']} | {row['activation_mode']} | {row['depth']} | "
            f"{float(row['test_at_best_val']):.4f} | {float(row['best_val']):.4f} | "
            f"`{row['snapped_widths']}` |"
        )

    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.append(
                f"- {row['activation_mode']}/{row['activation_kind']} depth {row['depth']} "
                f"on {row['device']}: {row['error']}"
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
    combos = activation_combinations(args.activation_modes, args.activation_kinds)

    for combo_idx, (activation_mode, activation_kind) in enumerate(combos, start=1):
        current_args = combo_args(args, activation_mode, activation_kind)
        combo_save_root = (
            save_root / f"mode_{activation_mode.replace('-', '_')}_kind_{activation_kind}"
        )
        for candidate in candidates:
            device = args.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            try:
                print(
                    "Training "
                    f"combo={combo_idx}/{len(combos)} "
                    f"mode={activation_mode} kind={activation_kind} "
                    f"depth={candidate.depth} widths={candidate.snapped_widths} on {device}...",
                    flush=True,
                )
                row = train_candidate(
                    candidate=candidate,
                    args=current_args,
                    data=data,
                    dataset=dataset,
                    device=device,
                    save_root=combo_save_root,
                )
            except RuntimeError as exc:
                if device == "cuda" and is_cuda_oom(exc):
                    torch.cuda.empty_cache()
                    print(
                        f"CUDA OOM for {activation_mode}/{activation_kind} "
                        f"depth={candidate.depth}; retrying on CPU...",
                        flush=True,
                    )
                    try:
                        row = train_candidate(
                            candidate=candidate,
                            args=current_args,
                            data=data,
                            dataset=dataset,
                            device="cpu",
                            save_root=combo_save_root,
                        )
                    except Exception as cpu_exc:
                        row = failed_row(candidate, "cpu", str(cpu_exc), current_args)
                else:
                    row = failed_row(candidate, device, str(exc), current_args)
            except Exception as exc:
                row = failed_row(candidate, device, str(exc), current_args)

            row["combo_id"] = combo_idx
            rows.append(row)
            write_outputs(result_dir, rows, args, save_root)

    print(f"Results written to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
