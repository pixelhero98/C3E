from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Implementations.c3e import ChanCapConEst  # noqa: E402
from Implementations.datasets import DATASET_CHOICES, dataset_slug, load_node_dataset  # noqa: E402
from Implementations.propanalyzer import (  # noqa: E402
    PropagationVarianceAnalyzer,
    apply_variance_guard,
)


DEFAULT_METHODS = ["gcn", "appnp", "gdc", "sgc", "chebnetii", "gprgnn", "jacobiconv", "s2gc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect C3E architecture candidates without training.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="Cora")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--eta", type=float, default=0.45)
    parser.add_argument("--max-layers", type=int, default=9)
    parser.add_argument("--variance-guard-ratio", type=float, default=0.95)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if args.eta <= 0:
        parser.error("--eta must be > 0.")
    if args.max_layers < 2:
        parser.error("--max-layers must be >= 2.")
    if not 0.0 <= args.variance_guard_ratio <= 1.0:
        parser.error("--variance-guard-ratio must be in [0.0, 1.0].")
    return args


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio = str(args.variance_guard_ratio).replace(".", "p")
    prefix = dataset_slug(args.dataset)
    return f"{prefix}_architectures_eta{str(args.eta).replace('.', 'p')}_vg{ratio}_{stamp}"


def preview(values: np.ndarray, limit: int = 5) -> str:
    return np.array2string(values[:limit], precision=6, separator=", ")


def inspect_method(
    method: str,
    data,
    H: float,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    analyzer = PropagationVarianceAnalyzer(data, method=method)
    raw_sigma = analyzer.compute_layerwise_variances(depth=args.max_layers)
    raw_sigma = np.maximum(raw_sigma, float(np.finfo(np.float64).eps))
    guarded_sigma = apply_variance_guard(raw_sigma, args.variance_guard_ratio)
    changed = int(np.sum(guarded_sigma > raw_sigma))
    estimator = ChanCapConEst(data=data, sigma_s=guarded_sigma, eta=args.eta)
    soft_width_guard = estimator.regularized_width_guard()

    try:
        layers, dropouts, capacities = estimator.optimize_weights(H=H, max_layers=args.max_layers)
    except Exception as exc:
        return [
            {
                "method": method,
                "status": "failed",
                "variance_guard_ratio": args.variance_guard_ratio,
                "depth": "",
                "widths": "",
                "mean_width": "",
                "max_width": "",
                "final_width": "",
                "phi0": "",
                "channel_capacity": "",
                "dropouts": "",
                "soft_width_guard": round(soft_width_guard, 6),
                "raw_sigma_preview": preview(raw_sigma),
                "guarded_sigma_preview": preview(guarded_sigma),
                "guard_changed_count": changed,
                "error": str(exc),
            }
        ]

    rows: list[dict[str, object]] = []
    for widths, dropout, capacity in zip(layers, dropouts, capacities):
        numeric_widths = [int(width) for width in widths]
        phi0 = estimator.constraint(np.asarray(numeric_widths, dtype=float), H) + H
        rows.append(
            {
                "method": method,
                "status": "success",
                "variance_guard_ratio": args.variance_guard_ratio,
                "depth": len(numeric_widths),
                "widths": numeric_widths,
                "mean_width": round(mean(numeric_widths), 3),
                "max_width": max(numeric_widths),
                "final_width": numeric_widths[-1],
                "phi0": round(float(phi0), 6),
                "channel_capacity": round(float(capacity), 6),
                "dropouts": [round(float(prob), 6) for prob in dropout],
                "soft_width_guard": round(soft_width_guard, 6),
                "raw_sigma_preview": preview(raw_sigma),
                "guarded_sigma_preview": preview(guarded_sigma),
                "guard_changed_count": changed,
                "error": "",
            }
        )
    return rows


def write_outputs(
    result_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    H: float,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "status",
        "variance_guard_ratio",
        "depth",
        "widths",
        "mean_width",
        "max_width",
        "final_width",
        "phi0",
        "channel_capacity",
        "dropouts",
        "soft_width_guard",
        "raw_sigma_preview",
        "guarded_sigma_preview",
        "guard_changed_count",
        "error",
    ]
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    valid_rows = [row for row in rows if row.get("status") == "success"]
    lines = [
        f"# {args.dataset} Architecture Candidates",
        "",
        "## Settings",
        "",
        f"- Dataset: {args.dataset}",
        f"- Eta: {args.eta}",
        f"- Valid window: [{H:.6f}, {H / args.eta:.6f}]",
        f"- Max layers: {args.max_layers}",
        f"- Variance guard ratio: {args.variance_guard_ratio}",
        f"- Data root: `{args.data_root}`",
        "",
        "## Valid Candidate Counts",
        "",
        "| Method | Count | Depths | Soft width guard | Guarded layers |",
        "| --- | ---: | --- | ---: | ---: |",
    ]

    for method in args.methods:
        method_rows = [row for row in valid_rows if row["method"] == method]
        failed_rows = [row for row in rows if row["method"] == method and row["status"] != "success"]
        if method_rows:
            depths = [int(row["depth"]) for row in method_rows]
            first = method_rows[0]
            lines.append(
                f"| `{method}` | {len(method_rows)} | {min(depths)}-{max(depths)} | "
                f"{float(first['soft_width_guard']):.3f} | {first['guard_changed_count']} |"
            )
        elif failed_rows:
            first = failed_rows[0]
            lines.append(
                f"| `{method}` | 0 | failed | {float(first['soft_width_guard']):.3f} | "
                f"{first['guard_changed_count']} |"
            )
        else:
            lines.append(f"| `{method}` | 0 | none |  |  |")

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Method | Depth | Widths | Phi0 | Capacity | Mean | Max | Final |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in valid_rows:
        lines.append(
            f"| `{row['method']}` | {row['depth']} | `{row['widths']}` | "
            f"{float(row['phi0']):.3f} | {float(row['channel_capacity']):.3f} | "
            f"{float(row['mean_width']):.1f} | {row['max_width']} | {row['final_width']} |"
        )

    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.append(f"- `{row['method']}`: {row['error']}")

    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = load_node_dataset(args.dataset, args.data_root)
    data = dataset[0]
    H = float(math.log(data.num_nodes))
    result_dir = args.results_root / run_name(args)

    rows: list[dict[str, object]] = []
    for method in args.methods:
        print(f"Inspecting {method}...", flush=True)
        rows.extend(inspect_method(method, data, H, args))
        write_outputs(result_dir, args, rows, H)

    print(f"Architecture inspection written to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
