import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root = 'Cora', '/home/PycharmProjects/pythonProject/data'
#================================================================================================================================================================
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
H = np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(k, C, H, True)
#================================================================================================================================================================
print(f'Current dataset is {dname}:')
#num_mmp_layer = [data.x.shape[1], 1982, 1300, 1092, 968, 766] [data.x.shape[1], 1982, 1308, 1080, 947, 726]
num_mmp_layer = [data.x.shape[1], 1982, 1298, 1094, 986, 722]
num_postpro_layer = [num_mmp_layer[-1], torch.unique(data.y).shape[0]]
num_skip_layer = num_mmp_layer[1:]
#p = [0.5803914072580636, 0.3961135669506727, 0.4568605721739575, 0.46962138534848274, 0.4415475075395707-0.1] [0.58, 0.395, 0.46, 0.47, 0.42] [0.586, 0.393, 0.456, 0.474, 0.4824]
p= [0.5803915091280454, 0.39768745419779084, 0.4522683210408116, 0.4670709288239129, 0.4342272413012144]

# Deep variant settings, i.e., simply stacking deeper with the width of original works
#num_mmp_layer = [data.x.shape[1], 16, 16, 16, 16, 16]
#num_postpro_layer = [16, torch.unique(data.y).shape[0]]
#num_skip_layer = [16, 16, 16, 16, 16]
#p = [0.6, 0.6, 0.6, 0.6, 0.6]
#num_mmp_layer = [data.x.shape[1], 64, 64, 64, 64, 64]
#num_postpro_layer = [64, torch.unique(data.y).shape[0]]
#num_skip_layer = [64, 64, 64, 64, 64]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9e-4, weight_decay=5e-4) #lr=0.9e-4, weight_decay=5e-4

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
