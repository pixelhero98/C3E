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

import random
import argparse
import sys
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from utility import create_masks 
from pathlib import Path
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
from Model_factory.model import Model
from c3e import ChanCapConEst
from propanalyzer import PropagationVarianceAnalyzer
from utility import train, val, test, save_checkpoint
from Visualizations.entropy_energy import representation_entropy, dirichlet_energy


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


def run_solution(data, dataset, layers: list, dropout: list, channel_capacity: float, args) -> None:
    prop_layer_sizes = layers
    drop_probs = dropout
    channel_str = f"{channel_capacity:.6g}".replace('.', 'p')
    sol_dir = args.save_dir / f"sol_{channel_str}"
    sol_dir.mkdir(exist_ok=True)

    model = Model(
        prop_layer=[data.x.shape[1]] + prop_layer_sizes,
        num_class=dataset.num_classes,
        drop_probs=drop_probs,
        use_activations=[True] * len(prop_layer_sizes),
        conv_methods=args.prop_method
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=20)

    best_val, best_test = 0.0, 0.0
    epochs_no_improve = 0
    logging.info(f"Starting training for solution: layers={prop_layer_sizes}, dropout={drop_probs}")

    try:
        for epoch in tqdm(range(1, args.epochs + 1), desc=f"Sol {channel_str}", file=sys.stdout):
            loss = train(model, data, optimizer)
            val_acc = val(model, data)
            test_acc = test(model, data)
            if test_acc > best_test:
                best_test = test_acc
                
            scheduler.step(val_acc)

            # Logging with learning rate
            if epoch % 10 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f}, val_acc: {val_acc:.4f}, lr: {current_lr:.2e}")

            # Early stopping & checkpointing
            if val_acc > best_val:
                best_val = val_acc
                epochs_no_improve = 0
                save_checkpoint(
                    sol_dir=sol_dir,
                    layer_str=channel_str,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val=best_val,
                    layer_sizes=prop_layer_sizes,
                    dropout=drop_probs
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logging.info(f"No improvement for {args.patience} epochs. Early stopping.")
                break
            
        logging.info(f"Solution:{prop_layer_sizes}, Network Channel Capacity:{channel_capacity}, best_val={best_val:.4f}, best_test_acc={best_test:.4f}")
        # print(f"Solution: Hidden dimensions:{prop_layer_sizes} / Dropout probabilities:{drop_probs}: best_val={best_val:.4f}, best_test={best_test:.4f}")

    except Exception as e:
        logging.error(f"Error in solution {channel_str}: {e}", exc_info=True)

    finally:
        # Clean up to free GPU memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        
    return best_test, sol_dir

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

    results, sol_dirs = [], []
    data = dataset[0]
    num_nodes = data.x.size(0)
    H = np.log(num_nodes)
    sigma_s = PropagationVarianceAnalyzer(data, method=args.prop_method).compute_variance()
    solutions = ChanCapConEst(data, args.eta, sigma_s).optimize_weights(H, verbose=True)

    if args.dataset not in {'ogbn-arxiv','ogbn-papers100M'}:
        data.train_mask, data.val_mask, data.test_mask = create_masks(data, args.train_per_class, args.num_val, args.num_test, seed=args.seed)
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
    # Iterate solutions safely
    for layers, dropout, channel_capacity in zip(solutions[0], solutions[1], solutions[-1]):
        res, sols = run_solution(data, dataset, layers, dropout, channel_capacity, args)
        results.append(res)
        sol_dirs.append(sols)

    opt_result_index = results.index(max(results))
    print(f"Optimal hidden dimensions:{solutions[0][opt_result_index]}, Dropout probabilities:{solutions[1][opt_result_index]}, Network Channel Capacity:{solutions[-1][opt_result_index]}, Performance:{max(results)}")
    rep_entropy = representation_entropy(sol_dirs[opt_result_index], args.device, solutions[0][opt_result_index], dataset.num_classes, solutions[1][opt_result_index], [True] * solutions[0][opt_result_index], args.prop_method, data, nbins=2000)
    rep_energy = dirichlet_energy(sol_dirs[opt_result_index], args.device, solutions[0][opt_result_index], dataset.num_classes, solutions[1][opt_result_index], [True] * solutions[0][opt_result_index], args.prop_method, data, normalized=True)
    
if __name__ == '__main__':
    main()
