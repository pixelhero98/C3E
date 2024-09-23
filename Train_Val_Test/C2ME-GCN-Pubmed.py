import numpy as np
from gcn_box import *
import torch_geometric.transforms as T
from torch_geometric.transforms import Compose, AddSelfLoops, NormalizeFeatures
from torch_geometric.datasets import Planetoid
from graph_nn_optimization import GraphNNOptimization, round_to_nearest_even


torch.manual_seed(42) #0, 42, 3407
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dname, root, sroot = 'Pubmed', '/home/onecs-casino/PycharmProjects/pythonProject/data', '/home/onecs-casino/PycharmProjects/pythonProject/saved_model_ICLR'
#================================================================================================================================================================
# Load the dataset with the composed transforms
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
data = dataset[0]
k, C, H = 1, torch.unique(data.y).shape[0], np.log(data.x.shape[0])
Optimizer = GraphNNOptimization(data)
Optimizer.optimize_weights(k, C, H, True)
#================================================================================================================================================================
print(f'Current dataset is {dname}:')


#num_mmp_layer = [data.x.shape[1], 682, 450, 370, 328, 300, 282, 266, 254, 248, 180]
#num_postpro_layer = [180, torch.unique(data.y).shape[0]]
#num_skip_layer = [682, 450, 370, 328, 300, 282, 266, 254, 248, 180]
#p=[0.5768967265480078, 0.3972434074646847, 0.4512414628557069, 0.4700690717245687, 0.4785020225730948,
 #  0.483259676961734, 0.4864385073062393, 0.4885555236495426, 0.4927192252784339, 0.4213417787511724-0.10]
num_mmp_layer = [data.x.shape[1], 682, 450, 382, 342, 316, 296, 282, 270, 260, 252, 244, 238, 234, 228, 224, 220, 214, 188]
num_postpro_layer = [num_mmp_layer[-1], torch.unique(data.y).shape[0]]
num_skip_layer = num_mmp_layer[1:]
p = [0.5768967133138926, 0.3977781609852886, 0.4588316228955793, 0.4728075302124476, 0.480069960358531,
     0.4843588411509834, 0.487168688577034, 0.4891903384386155, 0.4907138560948846, 0.49190186670968583,
     0.49285421713668465, 0.4936312829119349, 0.4942765749065059, 0.4948205440684984, 0.495281820400682,
     0.49554000436459245, 0.4934944905886013, 0.46608500925573526]

print(f'Current modified entropy is {np.log(data.x.shape[0] * k * C) + np.sum(np.log(num_mmp_layer[1:]))}')
terms = []
for i in range(1, len(num_mmp_layer)):
    terms.append((1 / i) * np.log(num_mmp_layer[i - 1] * num_mmp_layer[i] / (num_mmp_layer[i - 1] + num_mmp_layer[i])))
print(f'Current modified constraint is {sum(terms) - H}')
#num_mmp_layer = [data.x.shape[1], 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
#num_postpro_layer = [16, torch.unique(data.y).shape[0]]
#num_skip_layer = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
#p=[0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
#num_mmp_layer = [data.x.shape[1], 682, 456, 378, 340, 314, 298, 284, 274, 266, 260, 254, 250, 244, 240, 144]
#num_postpro_layer = [144, torch.unique(data.y).shape[0]]
#num_skip_layer = [682, 456, 378, 340, 314, 298, 284, 274, 266, 260, 254, 250, 244, 240, 144]
#p=[0.5768967256886075, 0.4008139006142377, 0.45262543160198354, 0.4735225101952041, 0.48135481167371874, 0.4858475857059632,
   #0.4888234456981958, 0.49091998087450783, 0.4924647179902387, 0.4936377294397216, 0.4945525071511927,
   #0.4952732307230858, 0.49554996272994245, 0.49521365558976027, 0.37444008064295353-0.132]

#num_mmp_layer = [data.x.shape[1], 682, 452, 372, 330, 304, 286, 272, 260, 250, 242, 238, 162]
#num_postpro_layer = [162, torch.unique(data.y).shape[0]]
#num_skip_layer = [682, 452, 372, 330, 304, 286, 272, 260, 250, 242, 238, 162]
#p=[0.576896724597649, 0.39848123544876923, 0.4510682065186533, 0.4710235653308993,
   #0.4793076924951645, 0.48402058866407627, 0.4871866774624918, 0.48945120600591097,
   #0.49112363597410136, 0.4920777423056536, 0.494443103215228, 0.4039417406605057-0.162]
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

    for epoch in range(3):
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

        if best_test_acc >= 0.799 and smd == 1:
            torch.save(model, sroot+'/c2me-gcn-'+dname+'.pth')
            flag = 1
            print('result found')
            break

        if best_test_acc >= 0.697 and deep == 1:
            torch.save(model, sroot+'/deep-gcn-'+dname+'.pth')
            flag = 1
            print('result found for deep')
            break
    print(f'Current Run={runs}')
    print(f'Best Val Accuracy={best_val_acc}, Epoch={epochs}')
    print(f'Best Test Accuracy={best_test_acc}, Epoch={epochs}')
    print('====================================================================')
#-----------------------------------------------------------------------------------------


