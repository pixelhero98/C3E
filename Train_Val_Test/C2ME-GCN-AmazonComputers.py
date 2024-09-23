import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


torch.manual_seed(42) #0, 42, 3407
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root, sroot = 'Computers', '/home/onecs-casino/PycharmProjects/pythonProject/data', '/home/onecs-casino/PycharmProjects/pythonProject/saved_model_ICLR'
#=================================================================================================================================================================
dataset = Amazon(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
k, C, H = 1, torch.unique(data.y).shape[0], np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(k, C, H, True)
#=================================================================================================================================================================
#num_mmp_layer = [data.x.shape[1], 846, 554, 464, 416, 386, 366, 292]
#num_postpro_layer = [292, torch.unique(data.y).shape[0]]
#num_skip_layer = [846, 554, 464, 416, 386, 366, 292]
#p = [0.5244952172987113, 0.3957914892880824, 0.4553490949133123, 0.47291842067105827, 0.4810080112578491, 0.48726138707336664, 0.44286318909224187-0.25]
#num_mmp_layer = [data.x.shape[1], 16, 16, 16, 16, 16, 16, 16]
#num_postpro_layer = [16, torch.unique(data.y).shape[0]]
#num_skip_layer = [16, 16, 16, 16, 16, 16, 16]
#p = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
num_mmp_layer = [data.x.shape[1], 846, 552, 469, 421, 389, 365, 347, 333, 320, 310, 302, 294, 287, 282, 276, 272, 267, 263, 257, 226]
num_postpro_layer = [num_mmp_layer[-1], torch.unique(data.y).shape[0]]
num_skip_layer = num_mmp_layer[1:]
p = [0.5244951655754977, 0.39512309114699873, 0.4594778474032184, 0.4728603680921133, 0.48021839167301505,
     0.4844655891655494, 0.4872508410240173, 0.48925621466740155, 0.490769182229684, 0.49194937650982695,
     0.4928937480821861, 0.49366762998770963, 0.4943068688871587, 0.49484972062763166, 0.4953112065339206,
     0.4957088573177959, 0.4960536168509915, 0.4962518121162641, 0.49439357932833006, 0.4674922806223869]
print(f'Current dataset is {dname}:')
print(f'Current modified entropy is {np.log(data.x.shape[0] * k * C) + np.sum(np.log(num_mmp_layer[1:]))}')
terms = []
for i in range(1, len(num_mmp_layer)):
    terms.append((1 / i) * np.log(num_mmp_layer[i - 1] * num_mmp_layer[i] / (num_mmp_layer[i - 1] + num_mmp_layer[i])))
print(f'Current modified constraint is {sum(terms) - H}')

num_classes = 10
train_per_class = 20
num_val = 1300
num_test = 11881
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
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-4)

    for epoch in range(350):
        loss = train(model, data, p, optimizer)
        valacc = val(model, data, p,)
        acc = test(model, data, p,)

        #print(f'Val Accuracy={valacc}, Test Accuracy={acc}, Train Loss={loss}, Epoch={epoch}')
        if valacc > best_val_acc:
            best_val_acc = valacc
            epochs_v = epoch

        if acc > best_test_acc:
            best_test_acc = acc
            epochs_t = epoch

        if best_test_acc >= 0.830 and smd == 1:
            torch.save(model, sroot+'/c2me-gcn-'+dname+'.pth')
            flag = 1
            break

        if best_test_acc >= 0.819 and deep == 1:
            torch.save(model, sroot+'/deep-gcn-'+dname+'.pth')
            flag = 1
            break
    print(f'Current Run={runs}')
    print(f'Best Val Accuracy={best_val_acc}, Epoch={epochs_v}')
    print(f'Best Test Accuracy={best_test_acc}, Epoch={epochs_t}')
    print('====================================================================')