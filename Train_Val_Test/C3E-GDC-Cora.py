import numpy as np
from gdc_box import *
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graph_nn_optimization import GraphNNOptimization


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root = 'Cora', '/home/PycharmProjects/pythonProject/data'
#================================================================================================================================================================
print(f'Current dataset is {dname}:')
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
# Using the default PPR kernel in GDC, heat kernel can be better but needs careful selection of diffusion time t and expansion steps.
transform=T.GDC(self_loop_weight=1,
                normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128, dim=0))
data = transform(data)
H = np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(H, True) 
# Round the continuous values to integer width.
#================================================================================================================================================================
num_mmp_layer = [data.x.shape[1], 1488, 972, 688]
num_postpro_layer = [num_mmp_layer[-1], torch.unique(data.y).shape[0]]
num_skip_layer = num_mmp_layer[1:]
p= [0.509475213987379, 0.3952697780106441, 0.4144606910003551]

# Deep variant settings, i.e., simply stacking deeper with the width of original works
#num_mmp_layer = [data.x.shape[1], 64, 64, 64]
#num_postpro_layer = [64, torch.unique(data.y).shape[0]]
#num_skip_layer = [64, 64, 64]
#p = [0.5, 0.5, 0.5] [0.6, 0.6, 0.6]
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4) #lr=0.9e-4, weight_decay=5e-4

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
