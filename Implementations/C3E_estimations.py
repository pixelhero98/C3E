import os
import random
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from Model_factory.model import Model
from c3e import ChanCapConEst
from propanalyzer import PropagationVarianceAnalyzer
from utility import train, val, test, save_model

# === Config & seeding ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_ROOT = '/home/.../data'
SAVE_DIR = '/home/.../saved'
os.makedirs(SAVE_DIR, exist_ok=True)

dname = 'Cora'
dataset = Planetoid(DATA_ROOT, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
H = np.log(data.x.shape[0])
eta = 0.5
prop_method = 'gcn'

# === Estimate good architectures ===
sigma_s = PropagationVarianceAnalyzer(data, method=prop_method)
estimator = ChanCapConEst(data, eta, sigma_s)
solutions = estimator.optimize_weights(H, verbose=True)

# === Training loop ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
NUM_EPOCH = 500
LR = 2e-4; WD = 5e-4

for sol in solutions:
    prop_layer_sizes, dropout = sol[0], sol[-1]
    model = Model(prop_layer=[data.x.shape[1]] + prop_layer_sizes,
                  num_class=dataset.num_classes,
                  drop_probs=dropout,
                  use_activations=[True]*len(prop_layer_sizes),
                  conv_methods=prop_method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
    
    best_val, best_test = 0.0, 0.0
    epochs_no_improve = 0
    for epoch in range(1, NUM_EPOCH+1):
        tr_loss = train(model, data, optimizer)
        val_acc = val(model, data)
        test_acc = test(model, data)
        
        # update scheduler
        scheduler.step(val_acc)
        
        # check for best
        if val_acc > best_val:
            best_val, epochs_no_improve = val_acc, 0
        else:
            epochs_no_improve += 1
        if test_acc > best_test:
            best_test = test_acc
            save_model(
                model, 
                os.path.join(SAVE_DIR, f'model_sol_{prop_layer_sizes}_ep{epoch}.pt')
            )
        # early stopping
        if epochs_no_improve >= 75:
            print(f"No improvement for 75 epochs; stopping.")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} — train loss {tr_loss:.4f}, val {val_acc:.4f}, test {test_acc:.4f}")
    
    print(f"Solution {prop_layer_sizes}: best val={best_val:.4f}, best test={best_test:.4f}")
