import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root = 'Photo', '/home/PycharmProjects/pythonProject/data'
#=================================================================================================================================================================
print(f'Current dataset is {dname}:')
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
H = np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(H, True) 
# Round the continuous values to integer width.
#=================================================================================================================================================================
num_mmp_layer = [data.x.shape[1], 830, 538, 446, 400, 370, 350, 274]
num_postpro_layer = [num_mmp_layer[-1], torch.unique(data.y).shape[0]]
num_skip_layer = num_mmp_layer[1:]
p = [0.5269704832368158, 0.393249463591657, 0.4537758691948178, 0.4722366163403161, 0.48017926635004293, 0.48640815762715806, 0.43972497525942633 -0.08]
num_classes = 8
train_per_class = 20
num_val = 1340
num_test = 5987
train_mask, val_mask, test_mask = create_balanced_masks(data.y, num_classes, train_per_class, num_val, num_test)
data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
            y=data.y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
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

    for epoch in range(350):
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
