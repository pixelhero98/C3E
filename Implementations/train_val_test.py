"""
Example usage:

    # Default settings on Cora (data in ~/data, outputs in ~/saved):
    python3 train_val_test.py

    # Using CiteSeer, custom paths, and tuned hyperparams:
    python3 train_val_test.py \
        --dataset CiteSeer \
        --data_root /path/to/data \
        --save_dir /path/to/checkpoints \
        --epochs 300 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --eta 0.7 \
        --patience 50 \
        --prop_method gcn \
        --seed 123 \
        --train_per_class 20 \
        --num_val 500 \
        --num_test 1000 \
        --device cuda
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon, Planetoid, WikipediaNetwork
from torch_geometric.data import Data
from tqdm import tqdm


# Ensure local modules (Implementations, Model_factory, Visualizations, ...) are importable
REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Model_factory.model import Model
from Visualizations.entropy_energy import dirichlet_energy, representation_entropy
from c3e import ChanCapConEst
from propanalyzer import PropagationVarianceAnalyzer
from utility import create_masks, save_checkpoint, test, train, val


CONV_METHOD_ALIASES = {
    'jacobiconv': 'jacobi',
}


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train GCN on Planetoid dataset with capacity estimation."
    )
    parser.add_argument('--data_root', type=Path, default=Path.home() / 'data',
                        help='Root directory for datasets')
    parser.add_argument('--save_dir', type=Path, default=Path.home() / 'saved',
                        help='Directory to save models and logs')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora','CiteSeer','PubMed','Chameleon','Squirrel','AmazonPhoto','AmazonComputers'],
                        help='Planetoid dataset name')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--eta', type=float, default=0.5,
                        help='Eta parameter for capacity estimator')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (in epochs)')
    parser.add_argument('--prop_method', type=str, default='gcn',
                        choices=['gcn', 'appnp', 'gdc', 'sgc',
                                 'chebnetii', 'gprgnn', 'jacobiconv', 's2gc'],
                        help='Propagation method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_per_class', type=int, default=20, help='Number of node per class')
    parser.add_argument('--num_val', type=int, default=500, help='Number of node for validation')
    parser.add_argument('--num_test', type=int, default=1000, help='Number of node for test')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()


def setup_logging(save_dir: Path) -> None:
    """Initialise file and console logging in ``save_dir``."""

    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=save_dir / 'training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def _format_layers(layers: Iterable[int]) -> str:
    return "-".join(str(int(l)) for l in layers)


def run_solution(
    data: Data,
    dataset,
    layers: Sequence[int],
    dropout: Sequence[float],
    channel_capacity: float,
    args: argparse.Namespace
) -> Tuple[float, Path]:
    """Train a candidate architecture and return the best test accuracy and checkpoint path."""

    prop_layer_sizes = list(layers)
    drop_probs = list(dropout)
    channel_str = f"{channel_capacity:.6g}".replace('.', 'p')
    sol_dir = args.save_dir / f"sol_{channel_str}"
    sol_dir.mkdir(parents=True, exist_ok=True)

    conv_method = CONV_METHOD_ALIASES.get(args.prop_method, args.prop_method)

    model = Model(
        prop_layer=[data.x.shape[1]] + prop_layer_sizes,
        num_class=dataset.num_classes,
        drop_probs=drop_probs,
        use_activations=[True] * len(prop_layer_sizes),
        conv_methods=conv_method
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=20
    )

    best_val, best_test = float("-inf"), 0.0
    best_checkpoint: Optional[Path] = None
    epochs_no_improve = 0
    logging.info("Starting training for solution: layers=%s, dropout=%s", prop_layer_sizes, drop_probs)

    try:
        for epoch in tqdm(range(1, args.epochs + 1), desc=f"Sol {channel_str}", file=sys.stdout):
            loss = train(model, data, optimizer)
            val_acc = val(model, data)
            test_acc = test(model, data)
            best_test = max(best_test, test_acc)

            scheduler.step(val_acc)

            if epoch % 10 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "Epoch %d/%d - loss: %.4f, val_acc: %.4f, lr: %.2e",
                    epoch,
                    args.epochs,
                    loss,
                    val_acc,
                    current_lr,
                )

            if val_acc > best_val:
                best_val = val_acc
                epochs_no_improve = 0
                best_checkpoint = save_checkpoint(
                    sol_dir=sol_dir,
                    layer_str=_format_layers(prop_layer_sizes),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val=best_val,
                    layer_sizes=prop_layer_sizes,
                    dropout=drop_probs,
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logging.info("No improvement for %d epochs. Early stopping.", args.patience)
                break

        logging.info(
            "Solution: %s | Channel Capacity: %.4f | best_val=%.4f | best_test=%.4f",
            prop_layer_sizes,
            channel_capacity,
            best_val,
            best_test,
        )

    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Error in solution %s: %s", channel_str, exc, exc_info=True)

    finally:
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_checkpoint is None:
        checkpoints = sorted(sol_dir.glob('best_val_*.pt'))
        if checkpoints:
            best_checkpoint = checkpoints[-1]
        else:
            raise RuntimeError(f"No checkpoints saved for solution {channel_str}.")

    return best_test, best_checkpoint

def main() -> None:
    args = parse_args()
    setup_logging(args.save_dir)
    logging.info(f"Arguments: {args}")
    set_seed(args.seed)

    if args.dataset in {'Cora','CiteSeer','PubMed'}:
        dataset = Planetoid(str(args.data_root), name=args.dataset, transform=T.AddSelfLoops())
    elif args.dataset in {'AmazonPhoto','AmazonComputers'}:
        dataset = Amazon(str(args.data_root), name=args.dataset, transform=T.AddSelfLoops())
    elif args.dataset in {'Chameleon','Squirrel'}:
        dataset = WikipediaNetwork(str(args.data_root), name=args.dataset, transform=T.AddSelfLoops())
    else:
        dataset = PygNodePropPredDataset(name=args.dataset, transform=T.AddSelfLoops())

    results: List[float] = []
    checkpoint_paths: List[Path] = []
    data = dataset[0]
    num_nodes = data.x.size(0)
    if num_nodes <= 0:
        logging.error("Dataset contains no nodes; cannot proceed with training.")
        return

    H = float(np.log(num_nodes))
    sigma_s = float(PropagationVarianceAnalyzer(data, method=args.prop_method).compute_variance())
    sigma_s = max(sigma_s, float(np.finfo(np.float64).eps))
    try:
        solutions = ChanCapConEst(data, args.eta, sigma_s).optimize_weights(H, verbose=True)
    except RuntimeError as exc:
        logging.error("Channel capacity optimisation failed: %s", exc)
        return

    if len(solutions) != 3:
        logging.error("Unexpected solution structure returned by optimiser: %s", solutions)
        return

    rounded_layers, dropout_schedules, channel_caps = solutions

    if not (rounded_layers and dropout_schedules and channel_caps):
        logging.error("Optimiser returned empty solution components: %s", solutions)
        return

    if not (len(rounded_layers) == len(dropout_schedules) == len(channel_caps)):
        logging.error(
            "Optimiser produced mismatched component lengths: %s", {
                'layers': len(rounded_layers),
                'dropout': len(dropout_schedules),
                'capacity': len(channel_caps),
            }
        )
        return

    if args.dataset not in {'ogbn-arxiv','ogbn-papers100M'}:
        data.train_mask, data.val_mask, data.test_mask = create_masks(
            data,
            args.train_per_class,
            args.num_val,
            args.num_test,
            seed=args.seed,
        )
    else:
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[valid_idx] = True
        data.test_mask[test_idx] = True
        
    data = data.to(args.device)
    for layers, dropout, channel_capacity in zip(rounded_layers, dropout_schedules, channel_caps):
        test_acc, checkpoint = run_solution(data, dataset, layers, dropout, channel_capacity, args)
        results.append(test_acc)
        checkpoint_paths.append(checkpoint)

    if not results:
        logging.error("No feasible solutions were trained.")
        return

    opt_result_index = max(range(len(results)), key=results.__getitem__)
    best_layers = rounded_layers[opt_result_index]
    best_dropouts = dropout_schedules[opt_result_index]
    best_capacity = channel_caps[opt_result_index]
    best_performance = results[opt_result_index]
    best_checkpoint = checkpoint_paths[opt_result_index]

    print(
        "Optimal hidden dimensions:{}, Dropout probabilities:{}, Network Channel Capacity:{:.4f}, Performance:{:.4f}".format(
            best_layers,
            best_dropouts,
            best_capacity,
            best_performance,
        )
    )

    use_activations = [True] * len(best_layers)
    prop_layers = [data.x.shape[1]] + best_layers

    rep_entropy = representation_entropy(
        best_checkpoint,
        args.device,
        prop_layers,
        dataset.num_classes,
        best_dropouts,
        use_activations,
        args.prop_method,
        data,
        nbins=2000,
    )
    rep_energy = dirichlet_energy(
        best_checkpoint,
        args.device,
        prop_layers,
        dataset.num_classes,
        best_dropouts,
        use_activations,
        args.prop_method,
        data,
        normalized=True,
    )

    logging.info("Representation entropy: %.6f", rep_entropy)
    logging.info("Dirichlet energy: %.6f", rep_energy)
    
if __name__ == '__main__':
    main()
