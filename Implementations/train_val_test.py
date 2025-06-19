"""
Example Usage:
    # Default settings (Cora dataset):
    python train_gcn.py

    # Custom dataset and paths:
    python train_val_test.py \
        --dataset CiteSeer \
        --data-root /path/to/data \
        --save-dir /path/to/checkpoints \
        --epochs 300 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --eta 0.7 \
        --patience 50 \
        --prop-method gcn \
        --seed 123

    # Minimal example (only specify save directory):
    python train_gcn.py --save-dir ./models
"""

import os
import random
import argparse
from pathlib import Path
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
from Model_factory.model import Model
from c3e import ChanCapConEst
from propanalyzer import PropagationVarianceAnalyzer
from utility import train, val, test

def set_seed(seed: int):
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


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train GCN on Planetoid dataset with capacity estimation."
    )
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'data', help='Root directory for datasets')
    parser.add_argument('--save-dir', type=Path, default=Path.home() / 'saved', help='Directory to save models and logs')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='Planetoid dataset name')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--eta', type=float, default=0.5, help='Eta parameter for capacity estimator')
    parser.add_argument('--patience', type=int, default=75, help='Early stopping patience (in epochs)')
    parser.add_argument('--prop-method', type=str, default='gcn', choices=['gcn', 'appnp', 'gdc', 'sgc',
    'chebnetii', 'gprgnn', 'jacobiconv', 's2gc'], help='Propagation method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    args.save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.save_dir / 'training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info(f"Arguments: {args}")

    # Reproducibility
    set_seed(args.seed)

    # Load dataset
    dataset = Planetoid(str(args.data_root), name=args.dataset, transform=T.AddSelfLoops())
    data = dataset[0]
    num_nodes = data.x.size(0)
    H = np.log(num_nodes)

    # Estimate architectures
    sigma_s = PropagationVarianceAnalyzer(data, method=args.prop_method)
    estimator = ChanCapConEst(data, args.eta, sigma_s)
    solutions = estimator.optimize_weights(H, verbose=True)

    # Move data to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Training loop over estimated solutions
    for layers, dropout in zip(solutions[0], solutions[1]):
        prop_layer_sizes = layers
        layer_str = '_'.join(map(str, prop_layer_sizes))
        sol_dir = args.save_dir / f"sol_{layer_str}"
        sol_dir.mkdir(exist_ok=True)

        model = Model(
            prop_layer=[data.x.shape[1]] + prop_layer_sizes,
            num_class=dataset.num_classes,
            drop_probs=dropout,
            use_activations=[True] * len(prop_layer_sizes),
            conv_methods=args.prop_method
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

        best_val = 0.0
        epochs_no_improve = 0

        logging.info(f"Starting training for solution: layers={prop_layer_sizes}, dropout={dropout}")
        for epoch in tqdm(range(1, args.epochs + 1), desc=f"Sol {layer_str}"):
            # Training step (handles model.train())
            loss = train(model, data, optimizer)

            # Validation (handles model.eval() and no_grad)
            val_acc = val(model, data)

            # Scheduler step
            scheduler.step(val_acc)

            # Early stopping and checkpointing
            if val_acc > best_val:
                best_val = val_acc
                epochs_no_improve = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val,
                    'layer_sizes': prop_layer_sizes,
                    'dropout': dropout,
                }
                ckpt_path = sol_dir / f"best_val_{layer_str}_ep{epoch}.pt"
                torch.save(checkpoint, ckpt_path)
                logging.info(f"Saved checkpoint: {ckpt_path}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logging.info(f"No improvement for {args.patience} epochs. Early stopping.")
                break

            if epoch % 10 == 0 or epoch == 1:
                logging.info(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f}, val_acc: {val_acc:.4f}")

        # Final test evaluation (handles eval and no_grad)
        test_acc = test(model, data)
        logging.info(f"Solution {prop_layer_sizes}: best_val={best_val:.4f}, test_acc={test_acc:.4f}")


if __name__ == '__main__':
    main()
