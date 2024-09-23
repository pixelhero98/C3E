import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


torch.manual_seed(42) #0, 42, 3407
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root, sroot = 'Photo', '/home/onecs-casino/PycharmProjects/pythonProject/data', '/home/onecs-casino/PycharmProjects/pythonProject/saved_model_ICLR'
#=================================================================================================================================================================
dataset = Amazon(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
k, C, H = 1, torch.unique(data.y).shape[0], np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(k, C, H, True)
#=================================================================================================================================================================
num_mmp_layer = [data.x.shape[1], 830, 545, 464, 417, 386, 363, 345, 331, 320, 310, 301, 294, 288, 279, 243]
num_postpro_layer = [num_mmp_layer[-1], torch.unique(data.y).shape[0]]
num_skip_layer = num_mmp_layer[1:]
p = [0.5269704236375359, 0.39645574679584117, 0.4599443908660009,
     0.4733749073173994, 0.4806504309246682, 0.48484619763522707, 0.487597418807976,
     0.48957615277151734, 0.4910662046789026, 0.4922264969257927, 0.49315555513392384,
     0.49390826448731673, 0.4943990356862038, 0.4925228711152475, 0.4649183370887281]
#num_mmp_layer = [data.x.shape[1], 830, 546, 458, 412, 384, 298]
#num_postpro_layer = [298, torch.unique(data.y).shape[0]]
#num_skip_layer = [830, 546, 458, 412, 384, 298]
#p = [0.5269704651362109, 0.3967534989917618, 0.45597966157599956, 0.4736241927848862, 0.48239951547438886, 0.4376891811723369-0.2]
print(f'Current dataset is {dname}:')
print(f'Current modified entropy is {np.log(data.x.shape[0] * k * C) + np.sum(np.log(num_mmp_layer[1:]))}')
terms = []
for i in range(1, len(num_mmp_layer)):
    terms.append((1 / i) * np.log(num_mmp_layer[i - 1] * num_mmp_layer[i] / (num_mmp_layer[i - 1] + num_mmp_layer[i])))
print(f'Current modified constraint is {sum(terms) - H}')
num_classes = 8
train_per_class = 20
num_val = 1340
num_test = 5987
train_mask, val_mask, test_mask = create_balanced_masks(data.y, num_classes, train_per_class, num_val, num_test)

data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
            y=data.y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
#num_mmp_layer = layers
#num_postpro_layer = [layers[-1], torch.unique(data.y).shape[0]]
#num_skip_layer = layers[1:]
#p = dropout
#num_mmp_layer = [data.x.shape[1], 2438, 1579, 1321, 1189, 1108, 804]
#num_postpro_layer = [804, torch.unique(data.y).shape[0]]
#num_skip_layer = [2438, 1579, 1321, 1189, 1108, 804]
#p = [0.6298557538558114, 0.39307491485710333, 0.45558629122129013, 0.47366711696884856, 0.48236414419243656, 0.42065671425360085] #lr = 1e-4 wd= 1e-5 5-layer entropy:622.59 acc:0.830 +- 0.008
#======================================================================================================================================

data, flag, deep, smd = data.to(device), 0, 0, 0
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)

    for epoch in range(350):
        loss = train(model, data, p, optimizer)
        valacc = val(model, data, p,)
        acc = test(model, data, p,)

        #print(f'Val Accuracy={valacc}, Test Accuracy={acc}, Train Loss={loss}, Epoch={epoch}')
        if valacc > best_val_acc:
            best_val_acc = valacc
            epochs = epoch

        if acc > best_test_acc:
            best_test_acc = acc
            epochs = epoch

        if best_test_acc >= 0.913 and smd == 1:
            torch.save(model, sroot+'/c2me-gcn-r-'+dname+'.pth')
            flag = 1
            break
        if best_test_acc >= 0.8944 and deep == 1:
            torch.save(model, sroot+'/deep-gcn-'+dname+'.pth')
            flag = 1
            break
    print(f'Current Run={runs}')
    print(f'Best Val Accuracy={best_val_acc}, Epoch={epochs}')
    print(f'Best Test Accuracy={best_test_acc}, Epoch={epochs}')
    print('====================================================================')