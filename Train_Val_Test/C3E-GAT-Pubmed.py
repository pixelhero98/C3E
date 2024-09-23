import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.transforms import Compose, AddSelfLoops, NormalizeFeatures
from torch_geometric.datasets import Planetoid
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root = 'Pubmed', '/home/PycharmProjects/pythonProject/data'
#================================================================================================================================================================
print(f'Current dataset is {dname}:')
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
H = np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(H, True) 
# Round the continuous values to integer width.
#================================================================================================================================================================
num_mmp_layer = [data.x.shape[1], int(682/2), 682, int(450/2), 450, int(370/2), 370, int(328/2), 328,
                 int(300/2), 300, int(282/2), 282, int(266/2), 266, int(254/2), 254,
                 int(248/2), 248, int(180/2), 180]
p = [0.5768967265480078, 0.3972434074646847, 0.4512414628557069, 0.4700690717245687, 0.4785020225730948,
   0.483259676961734, 0.4864385073062393, 0.4885555236495426, 0.4927192252784339, 0.4213417787511724-0.05]

# Deep variant settings, i.e., simply stacking deeper with the width of original works
#num_mmp_layer = [data.x.shape[1], 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
#num_postpro_layer = [64, torch.unique(data.y).shape[0]]
#num_skip_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
#p=[0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
#======================================================================================================================================

data, flag, deep = data.to(device), 0, 0
for runs in range(1):

    if flag == 1:
        break
        
    best_test_acc = 0
    best_val_acc = 0
    epochsv, epochst = 0, 0

    model = AAGNN(num_mmp_layer, num_postpro_layer, num_skip_layer)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    for epoch in range(90):
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
