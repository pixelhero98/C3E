import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root = 'Citeseer', '/home/PycharmProjects/pythonProject/data'
#================================================================================================================================================================
print(f'Current dataset is {dname}:')
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
H = np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(H, True) 
# Round the continuous values to integer width.
#================================================================================================================================================================
# Set head to 2, since 2 is easily divisible.
num_mmp_layer = [data.x.shape[1], int(5270/2), 5270, int(3432/2), 3432, int(2862/2), 2862, int(2126/2)]
num_skip_layer = [5270, 3432, 2862, 2126]
num_postpro_layer = [2126, torch.unique(data.y).shape[0]]
p = [0.5872912604291229, 0.3965438451299066, 0.45944052263339474, 0.4767243263290126, 0.41696622537165207 - 0.15]

# Deep variant settings, i.e., simply stacking deeper with the width of original works
# Set head to 8
#num_mmp_layer = [data.x.shape[1], 8, 8, 8, 8]
#num_postpro_layer = [64, torch.unique(data.y).shape[0]]
#num_skip_layer = [64, 64, 64, 64]
#p = [0.6, 0.6, 0.6, 0.6, 0.6]
#======================================================================================================================================
data, flag, deep = data.to(device), 0, 0
for runs in range(10):

    if flag == 1:
        break
        
    best_test_acc = 0
    best_val_acc = 0
    epochsv, epochst = 0, 0

    model = AAGNN(num_mmp_layer, num_postpro_layer, num_skip_layer)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)

    for epoch in range(200):
        loss = train(model, data, p, optimizer)
        valacc = val(model, data, p,)
        acc = test(model, data, p,)

        print(f'Val Accuracy={valacc}, Test Accuracy={acc}, Train Loss={loss}, Epoch={epoch}')
        if valacc > best_val_acc:
            best_val_acc = valacc
            epochsv = epoch

        if acc > best_test_acc:
            best_test_acc = acc
            epochst = epoch
    print(f'Current Run={runs}')
    print(f'Best Val Accuracy={best_val_acc}, Epoch={epochsv}')
    print(f'Best Test Accuracy={best_test_acc}, Epoch={epochst}')
    print('========================')
