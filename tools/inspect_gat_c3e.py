from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Implementations.c3e import ChanCapConEst  # noqa: E402
from Implementations.datasets import DATASET_CHOICES, dataset_slug, load_node_dataset  # noqa: E402
from Implementations.propanalyzer import apply_variance_guard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C3E architecture estimation using empirical GAT attention variance."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="Cora")
    parser.add_argument("--probe-summary", type=Path, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument(
        "--sigma-aggregate",
        choices=["geomean", "mean", "median"],
        default="geomean",
        help="How to aggregate best-checkpoint GAT sigmas when --sigma is not provided.",
    )
    parser.add_argument("--eta", type=float, default=0.45)
    parser.add_argument("--max-layers", type=int, default=9)
    parser.add_argument("--variance-guard-ratio", type=float, default=0.95)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if args.sigma is not None and args.sigma <= 0:
        parser.error("--sigma must be positive.")
    if args.eta <= 0:
        parser.error("--eta must be positive.")
    if args.max_layers < 2:
        parser.error("--max-layers must be >= 2.")
    if not 0.0 <= args.variance_guard_ratio <= 1.0:
        parser.error("--variance-guard-ratio must be in [0.0, 1.0].")
    return args


def latest_probe_summary(dataset: str) -> Path:
    prefix = dataset_slug(dataset)
    candidates = sorted(
        REPO_ROOT.glob(f"results/{prefix}_gat_variance_probe_*/summary.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No GAT variance probe summary found. Run tools/run_gat_variance_probe.py first."
        )
    return candidates[0]


def aggregate(values: list[float], mode: str) -> float:
    if not values:
        raise ValueError("No sigma values available for aggregation.")
    if mode == "mean":
        return float(mean(values))
    if mode == "median":
        return float(median(values))
    return float(math.exp(mean(math.log(value) for value in values)))


def load_sigma_from_probe(path: Path, aggregate_mode: str) -> tuple[float, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "success" and row.get("snapshot") == "best":
                rows.append(row)
    sigmas = [float(row["sigma_empirical"]) for row in rows]
    return aggregate(sigmas, aggregate_mode), rows


def run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = dataset_slug(args.dataset)
    return f"{prefix}_gat_c3e_eta{str(args.eta).replace('.', 'p')}_{stamp}"


def write_outputs(
    result_dir: Path,
    *,
    rows: list[dict[str, object]],
    calibration_rows: list[dict[str, str]],
    args: argparse.Namespace,
    sigma: float,
    sigma_source: str,
    H: float,
    width_guard: float,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "depth",
        "widths",
        "mean_width",
        "max_width",
        "final_width",
        "phi0",
        "channel_capacity",
        "dropouts",
        "sigma_empirical",
        "sigma_source",
        "variance_guard_ratio",
        "soft_width_guard",
    ]
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    with (result_dir / "calibration_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        if calibration_rows:
            writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibration_rows)

    lines = [
        f"# {args.dataset} GAT C3E Estimation",
        "",
        "## Settings",
        "",
        f"- Dataset: {args.dataset}",
        f"- Eta: {args.eta}",
        f"- Valid window: [{H:.6f}, {H / args.eta:.6f}]",
        f"- Max layers: {args.max_layers}",
        f"- Empirical sigma: {sigma:.8e}",
        f"- Sigma source: `{sigma_source}`",
        f"- Variance guard ratio: {args.variance_guard_ratio}",
        f"- Soft width guard: {width_guard:.3f}",
        "",
        "## Calibration Rows",
        "",
        "| Width | Heads | Channels/head | Val | Test | Sigma empirical |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in calibration_rows:
        lines.append(
            f"| {row.get('width', '')} | {row.get('heads', '')} | "
            f"{row.get('channels_per_head', '')} | {float(row.get('val_acc', 0.0)):.4f} | "
            f"{float(row.get('test_acc', 0.0)):.4f} | "
            f"{float(row.get('sigma_empirical', 0.0)):.8e} |"
        )

    lines.extend(
        [
            "",
            "## Estimated Candidates",
            "",
            "| Depth | Widths | Phi0 | Capacity | Mean | Max | Final |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['depth']} | `{row['widths']}` | {float(row['phi0']):.3f} | "
            f"{float(row['channel_capacity']):.3f} | {float(row['mean_width']):.1f} | "
            f"{row['max_width']} | {row['final_width']} |"
        )

    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    calibration_rows: list[dict[str, str]] = []
    sigma_source: str
    if args.sigma is not None:
        sigma = float(args.sigma)
        sigma_source = "manual --sigma"
    else:
        probe_summary = args.probe_summary or latest_probe_summary(args.dataset)
        sigma, calibration_rows = load_sigma_from_probe(probe_summary, args.sigma_aggregate)
        sigma_source = f"{args.sigma_aggregate} best sigma from {probe_summary}"

    dataset = load_node_dataset(args.dataset, args.data_root)
    data = dataset[0]
    H = float(math.log(data.num_nodes))
    raw_sigma = np.full(args.max_layers, sigma, dtype=float)
    guarded_sigma = apply_variance_guard(raw_sigma, args.variance_guard_ratio)
    estimator = ChanCapConEst(data=data, sigma_s=guarded_sigma, eta=args.eta)
    width_guard = estimator.regularized_width_guard()
    layers, dropouts, capacities = estimator.optimize_weights(H=H, max_layers=args.max_layers)

    rows: list[dict[str, object]] = []
    for widths, dropout, capacity in zip(layers, dropouts, capacities):
        numeric_widths = [int(width) for width in widths]
        phi0 = estimator.constraint(np.asarray(numeric_widths, dtype=float), H) + H
        rows.append(
            {
                "method": "gat_empirical",
                "depth": len(numeric_widths),
                "widths": numeric_widths,
                "mean_width": round(mean(numeric_widths), 3),
                "max_width": max(numeric_widths),
                "final_width": numeric_widths[-1],
                "phi0": round(float(phi0), 6),
                "channel_capacity": round(float(capacity), 6),
                "dropouts": [round(float(prob), 6) for prob in dropout],
                "sigma_empirical": sigma,
                "sigma_source": sigma_source,
                "variance_guard_ratio": args.variance_guard_ratio,
                "soft_width_guard": round(width_guard, 6),
            }
        )

    result_dir = args.results_root / run_name(args)
    write_outputs(
        result_dir,
        rows=rows,
        calibration_rows=calibration_rows,
        args=args,
        sigma=sigma,
        sigma_source=sigma_source,
        H=H,
        width_guard=width_guard,
    )
    print(f"GAT C3E estimation written to {result_dir}", flush=True)


if __name__ == "__main__":
    main()
