# C3E: Channel Capacity Constrained Estimation

C3E estimates candidate graph neural network architectures by matching channel
capacity constraints before training. The current code supports spectral graph
operators analytically and GAT-style attention message passing through an
empirical variance calibration workflow.

Paper draft: https://arxiv.org/abs/2511.06443

## Installation

Use Python 3.8 or newer. Install PyTorch and PyTorch Geometric wheels that match
your CUDA or CPU environment, then install the project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

For CUDA environments, install the PyG extension wheels from the wheel index
matching your PyTorch/CUDA version before running experiments.

```bash
python -m pip install torch-geometric torch-sparse ogb
```

All commands below are intended to run from the repository root. Datasets are
stored in `data/`, checkpoints in `saved/`, and summaries in `results/` by
default. These generated directories are ignored by git.

## Spectral C3E Optimization

The spectral path precomputes propagation variance analytically, estimates all
valid C3E candidates inside the window `[H, H / eta]`, and trains those
candidates with residual projections after the first propagation layer.

Supported spectral graph operations:

```text
gcn, appnp, gdc, sgc, chebnetii, gprgnn, jacobiconv, s2gc
```

Run one operation:

```bash
python -m Implementations.train_val_test \
  --dataset Cora \
  --prop_method gcn \
  --eta 0.45 \
  --max_layers 9 \
  --epochs 50 \
  --patience 10 \
  --device cuda
```

Important defaults:

- `--variance_guard_ratio 0.95` floors each layer variance to at least `0.95`
  of the previous layer before estimation. Pass `0.0` to disable it.
- `--activation_mode first-on` applies activation only after the first
  propagation layer. Aliases are accepted: `first-only`, `all`, and `none`.
- `--activation_kind prelu` can be switched to `silu` or `gelu`.
- Residual projections are used after layer 1 for estimated deep candidates.

Run an all-operation Cora screen:

```bash
python tools/run_cora_eta045_all_ops.py \
  --eta 0.45 \
  --max-layers 9 \
  --epochs 20 \
  --patience 5 \
  --device cuda
```

The batch runner writes one result folder under `results/` and one checkpoint
folder under `saved/`, records failed methods instead of stopping the whole run,
and keeps only the best checkpoint for each successful graph operation.

## Attention-Based MPNN / GAT Workflow

For broad message-passing networks, C3E first needs an empirical propagation
variance estimate. GAT v1 is the first supported attention-based path. It trains
one-hidden-layer GAT probes at multiple power-of-two total widths, estimates the
variance of learned attention matrices, and then feeds that variance into the
C3E estimator.

1. Probe empirical GAT attention variance:

```bash
python tools/run_cora_gat_variance_probe.py \
  --widths 16 64 128 \
  --heads 2 \
  --epochs 50 \
  --patience 10 \
  --device cuda
```

2. Estimate C3E GAT candidates from the probe:

```bash
python tools/inspect_cora_gat_c3e.py \
  --eta 0.45 \
  --max-layers 9 \
  --variance-guard-ratio 0.95
```

If `--probe-summary` is omitted, the estimator uses the newest
`results/cora_gat_variance_probe_*/summary.csv`.

3. Train the estimated residual GAT candidates:

```bash
python tools/run_cora_gat_c3e_candidates.py \
  --heads 2 \
  --epochs 50 \
  --patience 10 \
  --lr 1e-4 \
  --activation-mode first-on \
  --activation-kind prelu \
  --device cuda
```

If `--candidate-summary` is omitted, the trainer uses the newest
`results/cora_gat_c3e_*/summary.csv`.

4. Run the full activation search:

```bash
python tools/run_cora_gat_activation_grid.py \
  --heads 2 \
  --epochs 50 \
  --patience 10 \
  --lr 1e-4 \
  --device cuda
```

This expands every C3E candidate across:

```text
3 activation modes  x  3 activation kinds
first-on/all-on/all-off x prelu/silu/gelu
```

Candidate checkpoints are selected by best validation accuracy, and the final
summary ranks candidates by test accuracy at the validation-best checkpoint.
The smaller default GAT candidate learning rate is intentional: very wide or
deep estimated GATs were more stable around `0.9e-4` to `2e-4` than with larger
rates.

GraphSAGE, GIN, and other MPNNs are not part of this release. They can follow
the same empirical-variance pattern once an appropriate propagation-variance
estimator is added.

## Development Checks

```bash
python -m ruff check . --exclude data --exclude saved --exclude results
python -m pytest tests
python -m Implementations.train_val_test --help
python tools/run_cora_gat_c3e_candidates.py --help
python tools/run_cora_gat_activation_grid.py --help
```

The test suite covers variance guarding, C3E candidate filtering, activation
normalization, residual model construction, GAT empirical variance helpers, and
the activation-grid expansion.
