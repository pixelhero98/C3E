import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


torch.manual_seed(42) #0, 42, 3407
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root, sroot = 'Citeseer', '/home/onecs-casino/PycharmProjects/pythonProject/data', '/home/onecs-casino/PycharmProjects/pythonProject/saved_model_ICLR'
#================================================================================================================================================================
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
k, C, H = 1, torch.unique(data.y).shape[0], np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(k, C, H, True)
#================================================================================================================================================================
print(f'Current dataset is {dname}:')
#num_mmp_layer = [data.x.shape[1], 5270, 3462, 2944, 2682, 1918]
#num_postpro_layer = [1918, torch.unique(data.y).shape[0]]
#num_skip_layer = [5270, 3462, 2944, 2682, 1918]
#p = [0.5872912604291229, 0.3965438451299066, 0.45944052263339474, 0.4767243263290126, 0.41696622537165207 - 0.06]
#num_mmp_layer = [data.x.shape[1], 5270, 3427, 2878, 2546, 512] #2398
#num_postpro_layer = [512, torch.unique(data.y).shape[0]]
#num_skip_layer = [5270, 3427, 2878, 2546, 512]
#p = [0.5872911628220947, 0.39408770047540564, 0.4563656675822564, 0.4695390080463263, 0.44117897849028]
num_mmp_layer = [data.x.shape[1], 16, 16, 16, 16]
num_postpro_layer = [16, torch.unique(data.y).shape[0]]
num_skip_layer = [16, 16, 16, 16]
p = [0.6, 0.6, 0.6, 0.6]
print(f'Current modified entropy is {np.log(data.x.shape[0] * k * C) + np.sum(np.log(num_mmp_layer[1:]))}')
terms = []
for i in range(1, len(num_mmp_layer)):
    terms.append(
        (1 / i) * np.log(num_mmp_layer[i - 1] * num_mmp_layer[i] / (num_mmp_layer[i - 1] + num_mmp_layer[i])))
print(f'Current modified constraint is {sum(terms) - H}')
pc = []
for idx in range(len(num_mmp_layer) - 1):
    w0, w1 = num_mmp_layer[idx], num_mmp_layer[idx + 1]
    c = (w0 * w1) / (w0 + w1)
    pc.append(1 - c / w1)
print(pc)

#======================================================================================================================================

data, flag, deep, save_mod = data.to(device), 0, 0, 0
for runs in range(1):

    # if len(avg) == 10:
    # break
    if flag == 1:
        break
    best_test_acc = 0
    best_val_acc = 0
    epochs = 0
    flag = 0


    model = AAGNN(num_mmp_layer, num_postpro_layer, num_skip_layer)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4) #lr=5e-4, weight_decay=5e-4

    for epoch in range(800):
        loss = train(model, data, p, optimizer)
        valacc = val(model, data, p,)
        acc = test(model, data, p,)

        print(f'Val Accuracy={valacc}, Test Accuracy={acc}, Train Loss={loss}, Epoch={epoch}')
        if valacc > best_val_acc:
            best_val_acc = valacc
            epochs = epoch

        if acc > best_test_acc:
            best_test_acc = acc
            epochs = epoch

        if best_test_acc >= 0.735 and save_mod == 1:
            torch.save(model, sroot+'/c2me-gcn-'+dname+'.pth')
            flag = 1
            break
        if best_test_acc >= 0.682 and deep == 1:
            torch.save(model, sroot+'/deep-gcn-'+dname+'.pth')
            flag = 1
            break
    print(f'Current Run={runs}')
    print(f'Best Val Accuracy={best_val_acc}, Epoch={epochs}')
    print(f'Best Test Accuracy={best_test_acc}, Epoch={epochs}')
    print('====================================================================')



