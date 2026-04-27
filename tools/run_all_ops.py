from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_factory.activations import (  # noqa: E402
    ACTIVATION_KINDS,
    ACTIVATION_MODE_CHOICES,
    normalize_activation_mode,
)
from Implementations.datasets import DATASET_CHOICES, dataset_slug  # noqa: E402

DEFAULT_METHODS = ["gcn", "appnp", "gdc", "sgc", "chebnetii", "gprgnn", "jacobiconv", "s2gc"]
START_RE = re.compile(r"Starting training for solution: layers=(\[.*?\]), dropout=(\[.*?\])")
SOLUTION_RE = re.compile(
    r"Solution: (?P<layers>\[.*?\]) \| Channel Capacity: (?P<capacity>[-+0-9.eE]+) "
    r"\| best_val=(?P<best_val>[-+0-9.eE]+) \| test@best_val=(?P<test>[-+0-9.eE]+)"
)
BEST_RE = re.compile(
    r"Best solution index=(?P<index>\d+) \| layers=(?P<layers>\[.*?\]) "
    r"\| dropouts=(?P<dropouts>\[.*?\]) \| test@best_val=(?P<test>[-+0-9.eE]+) "
    r"\| checkpoint=(?P<checkpoint>.*)"
)


@dataclass
class MethodRun:
    method: str
    status: str
    device: str
    save_dir: Path
    returncode: int
    stdout_path: Path
    stderr_path: Path
    error: str = ""
    retained_checkpoint: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C3E architecture screen across graph ops.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="Cora")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--eta", type=float, default=0.45)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-layers", type=int, default=9)
    parser.add_argument("--variance-guard-ratio", type=float, default=0.95)
    parser.add_argument(
        "--activation-mode",
        choices=ACTIVATION_MODE_CHOICES,
        default="first-on",
    )
    parser.add_argument("--activation-kind", choices=ACTIVATION_KINDS, default="prelu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--saved-root", type=Path, default=REPO_ROOT / "saved")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--keep-best-only", action="store_true", default=True)
    args = parser.parse_args()
    if not 0.0 <= args.variance_guard_ratio <= 1.0:
        parser.error("--variance-guard-ratio must be in [0.0, 1.0].")
    args.activation_mode = normalize_activation_mode(args.activation_mode)
    args.activation_kind = args.activation_kind.strip().lower()
    return args


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = dataset_slug(args.dataset)
    eta = str(args.eta).replace(".", "p")
    return f"{prefix}_eta{eta}_all_ops_{stamp}"


def command_for(args: argparse.Namespace, method: str, save_dir: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "Implementations.train_val_test",
        "--dataset",
        args.dataset,
        "--prop_method",
        method,
        "--data_root",
        str(args.data_root),
        "--save_dir",
        str(save_dir),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--max_layers",
        str(args.max_layers),
        "--eta",
        str(args.eta),
        "--variance_guard_ratio",
        str(args.variance_guard_ratio),
        "--activation_mode",
        args.activation_mode,
        "--activation_kind",
        args.activation_kind,
        "--seed",
        str(args.seed),
        "--device",
        device,
    ]


def is_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    return "cuda" in lowered and "out of memory" in lowered


def run_method(args: argparse.Namespace, method: str, save_root: Path) -> MethodRun:
    method_dir = save_root / method
    method_dir.mkdir(parents=True, exist_ok=True)

    def attempt(device: str) -> subprocess.CompletedProcess[str]:
        cmd = command_for(args, method, method_dir, device)
        (method_dir / f"command_{device}.txt").write_text(" ".join(cmd), encoding="utf-8")
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (method_dir / f"stdout_{device}.txt").write_text(result.stdout, encoding="utf-8")
        (method_dir / f"stderr_{device}.txt").write_text(result.stderr, encoding="utf-8")
        return result

    result = attempt(args.device)
    device_used = args.device
    stdout_path = method_dir / f"stdout_{device_used}.txt"
    stderr_path = method_dir / f"stderr_{device_used}.txt"
    combined = f"{result.stdout}\n{result.stderr}"

    if result.returncode != 0 and args.device == "cuda" and is_cuda_oom(combined):
        result = attempt("cpu")
        device_used = "cpu"
        stdout_path = method_dir / "stdout_cpu.txt"
        stderr_path = method_dir / "stderr_cpu.txt"
        combined = f"{result.stdout}\n{result.stderr}"

    status = "success" if result.returncode == 0 else "failed"
    error = "" if status == "success" else combined[-4000:]
    return MethodRun(
        method=method,
        status=status,
        device=device_used,
        save_dir=method_dir,
        returncode=result.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        error=error,
    )


def parse_list(text: str) -> list[float]:
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        raise ValueError(f"Expected list, got: {text}")
    return value


def parse_training_log(run: MethodRun) -> tuple[list[dict[str, object]], dict[str, object]]:
    log_path = run.save_dir / "training.log"
    candidates: list[dict[str, object]] = []
    best: dict[str, object] = {}
    pending: list[dict[str, object]] = []

    if not log_path.exists():
        return candidates, best

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = START_RE.search(line)
        if start:
            layers = parse_list(start.group(1))
            dropouts = parse_list(start.group(2))
            pending.append({"layers": layers, "dropouts": dropouts})
            continue

        solution = SOLUTION_RE.search(line)
        if solution:
            layers = parse_list(solution.group("layers"))
            pending_item = pending.pop(0) if pending else {"layers": layers, "dropouts": []}
            widths = [int(width) for width in pending_item.get("layers", layers)]
            dropouts = [float(prob) for prob in pending_item.get("dropouts", [])]
            candidates.append(
                {
                    "method": run.method,
                    "status": run.status,
                    "device": run.device,
                    "depth": len(widths),
                    "widths": widths,
                    "mean_width": round(mean(widths), 3) if widths else math.nan,
                    "max_width": max(widths) if widths else math.nan,
                    "final_width": widths[-1] if widths else math.nan,
                    "dropouts": dropouts,
                    "channel_capacity": float(solution.group("capacity")),
                    "best_val": float(solution.group("best_val")),
                    "test_at_best_val": float(solution.group("test")),
                    "save_dir": str(run.save_dir),
                    "retained_checkpoint": run.retained_checkpoint,
                    "error": run.error,
                }
            )
            continue

        best_match = BEST_RE.search(line)
        if best_match:
            best = {
                "method": run.method,
                "best_index": int(best_match.group("index")),
                "layers": parse_list(best_match.group("layers")),
                "dropouts": parse_list(best_match.group("dropouts")),
                "test_at_best_val": float(best_match.group("test")),
                "checkpoint": best_match.group("checkpoint").strip(),
            }

    if not candidates and run.status != "success":
        candidates.append(
            {
                "method": run.method,
                "status": run.status,
                "device": run.device,
                "depth": "",
                "widths": "",
                "mean_width": "",
                "max_width": "",
                "final_width": "",
                "dropouts": "",
                "channel_capacity": "",
                "best_val": "",
                "test_at_best_val": "",
                "save_dir": str(run.save_dir),
                "retained_checkpoint": "",
                "error": run.error,
            }
        )

    return candidates, best


def prune_checkpoints(run: MethodRun, best: dict[str, object]) -> str:
    checkpoints = sorted(run.save_dir.rglob("*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        return ""

    keep = Path(str(best.get("checkpoint", ""))) if best.get("checkpoint") else checkpoints[-1]
    if not keep.is_absolute():
        keep = (REPO_ROOT / keep).resolve()
    else:
        keep = keep.resolve()

    save_dir = run.save_dir.resolve()
    try:
        keep.relative_to(save_dir)
    except ValueError:
        keep = checkpoints[-1].resolve()

    for checkpoint in checkpoints:
        resolved = checkpoint.resolve()
        try:
            resolved.relative_to(save_dir)
        except ValueError:
            continue
        if resolved != keep and resolved.exists():
            resolved.unlink()

    return str(keep)


def format_float(value: object) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def summarize_error(error: str) -> str:
    interesting = [
        line.strip()
        for line in error.splitlines()
        if "Capacity estimator failed" in line
        or "RuntimeError:" in line
        or "AssertionError" in line
        or "out of memory" in line.lower()
    ]
    if interesting:
        return interesting[0]
    stripped = " ".join(error.split())
    return stripped[:240]


def trend_note(method: str, rows: list[dict[str, object]]) -> str:
    valid = [row for row in rows if row.get("test_at_best_val") not in ("", None)]
    if not valid:
        return f"- `{method}` did not produce successful candidate metrics."
    best = max(valid, key=lambda row: float(row["test_at_best_val"]))
    shallow = min(valid, key=lambda row: int(row["depth"]))
    deep = max(valid, key=lambda row: int(row["depth"]))
    direction = "flat"
    if float(deep["test_at_best_val"]) > float(shallow["test_at_best_val"]) + 0.02:
        direction = "improved with depth"
    elif float(deep["test_at_best_val"]) < float(shallow["test_at_best_val"]) - 0.02:
        direction = "dropped with depth"
    return (
        f"- `{method}` best test {format_float(best['test_at_best_val'])} at depth "
        f"{best['depth']} with widths {best['widths']}; shallow-to-deep trend {direction}."
    )


def write_outputs(
    args: argparse.Namespace,
    result_dir: Path,
    runs: list[MethodRun],
    rows: list[dict[str, object]],
    best_by_method: dict[str, dict[str, object]],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "status",
        "device",
        "depth",
        "widths",
        "mean_width",
        "max_width",
        "final_width",
        "dropouts",
        "channel_capacity",
        "best_val",
        "test_at_best_val",
        "save_dir",
        "retained_checkpoint",
        "error",
    ]
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    successful_rows = [row for row in rows if row.get("status") == "success"]
    best_overall = (
        max(successful_rows, key=lambda row: float(row["test_at_best_val"]))
        if successful_rows
        else None
    )

    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    lines = [
        f"# {args.dataset} eta={args.eta} All-Operations Architecture Screen",
        "",
        "## Settings",
        "",
        f"- Dataset: {args.dataset}",
        f"- Methods: {', '.join(args.methods)}",
        f"- Seed: {args.seed}",
        f"- Epochs/patience/max_layers: {args.epochs}/{args.patience}/{args.max_layers}",
        f"- Variance guard ratio: {args.variance_guard_ratio}",
        f"- Activation mode/kind: {args.activation_mode}/{args.activation_kind}",
        f"- Requested device: {args.device}",
        f"- Data root: `{args.data_root}`",
        "",
        "## Status",
        "",
        "| Method | Status | Device | Retained checkpoint |",
        "| --- | --- | --- | --- |",
    ]
    for run in runs:
        retained = run.retained_checkpoint or ""
        lines.append(f"| `{run.method}` | {run.status} | {run.device} | `{retained}` |")

    failed = [run for run in runs if run.status != "success"]
    if failed:
        lines.extend(["", "## Failure Notes", ""])
        for run in failed:
            lines.append(f"- `{run.method}`: {summarize_error(run.error)}")

    lines.extend(["", "## Best Overall", ""])
    if best_overall:
        lines.append(
            f"`{best_overall['method']}` reached test@best_val "
            f"{format_float(best_overall['test_at_best_val'])} at depth "
            f"{best_overall['depth']} with widths `{best_overall['widths']}`."
        )
    else:
        lines.append("No method completed successfully.")

    lines.extend(
        [
            "",
            "## Best Per Method",
            "",
            "| Method | Depth | Widths | Best val | Test@best val |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for method in args.methods:
        valid = [
            row
            for row in by_method.get(method, [])
            if row.get("status") == "success" and row.get("test_at_best_val") not in ("", None)
        ]
        if not valid:
            lines.append(f"| `{method}` |  |  |  |  |")
            continue
        best = max(valid, key=lambda row: float(row["test_at_best_val"]))
        lines.append(
            f"| `{method}` | {best['depth']} | `{best['widths']}` | "
            f"{format_float(best['best_val'])} | {format_float(best['test_at_best_val'])} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Architectures",
            "",
            "| Method | Depth | Widths | Mean width | Max width | Final width | Capacity | Best val | Test@best val |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in successful_rows:
        lines.append(
            f"| `{row['method']}` | {row['depth']} | `{row['widths']}` | "
            f"{format_float(row['mean_width'])} | {row['max_width']} | {row['final_width']} | "
            f"{format_float(row['channel_capacity'])} | {format_float(row['best_val'])} | "
            f"{format_float(row['test_at_best_val'])} |"
        )

    lines.extend(["", "## Depth/Width Trends", ""])
    for method in args.methods:
        lines.append(trend_note(method, by_method.get(method, [])))

    if best_by_method:
        lines.extend(["", "## Logged Best Checkpoints", ""])
        for method, best in best_by_method.items():
            lines.append(f"- `{method}`: `{best.get('checkpoint', '')}`")

    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    name = run_name(args)
    save_root = args.saved_root / name
    result_dir = args.results_root / name
    save_root.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)

    runs: list[MethodRun] = []
    all_rows: list[dict[str, object]] = []
    best_by_method: dict[str, dict[str, object]] = {}

    for method in args.methods:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running {method}...", flush=True)
        run = run_method(args, method, save_root)
        rows, best = parse_training_log(run)
        if args.keep_best_only and best:
            run.retained_checkpoint = prune_checkpoints(run, best)
            for row in rows:
                row["retained_checkpoint"] = run.retained_checkpoint
        if best:
            best_by_method[method] = best
        runs.append(run)
        all_rows.extend(rows)
        write_outputs(args, result_dir, runs, all_rows, best_by_method)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] {method}: {run.status} on {run.device}",
            flush=True,
        )

    print(f"Results written to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
