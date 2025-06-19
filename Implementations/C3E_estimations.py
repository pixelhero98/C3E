import torch_geometric.transforms as T
from Model_factory.model import Model
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork
from c3e import ChanCapConEst
from propanalyzer import PropagationVarianceAnalyzer
from utility import train, val, test, save_model


dname, root, store = 'Cora', '/home/PycharmProjects/pythonProject/data', '/home/PycharmProjects/pythonProject/saved'
dataset = Planetoid(root, name=dname, transform=T.AddSelfLoops())
H, data, num_classes, eta = np.log(dataset[0].x.shape[0]), dataset[0], dataset.num_classes, 0.5

prop_method = 'gcn'
sigma_s = PropagationVarianceAnalyzer(data, method=prop_method)
Estimator = ChanCapConEst(data, 0.5, sigma_s)
solutions = Estimator.optimize_weights(H, True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 500
data = data.to(device)

for sol in solutions:
  prop_layer = [data.x.shape[1]] + sol[0]
  num_class = num_classes
  dropout = sol[-1]
  activation = [True] * len(sol[0])
  model = Model(prop_layer, num_class, dropout, activation, prop_method)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
  model = model.to(device)

  best_val_perf, best_test_perf, best_val_epoch, best_test_epoch = 0, 0, 0, 0

  for epoch in range(num_epoch):
    tr_loss = train(model, data, optimizer)
    val_perf = val(model, data)
    test_perf = test(model, data)
    
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        best_val_epoch = epoch
      
    if test_perf > best_test_perf:
        best_test_perf = test_perf
        best_test_epoch = epoch
        save_model(model, store)
        
  print(f'Current Solution={sol[0]}: Best Val Accuracy={best_val_perf}, Epoch={best_val_epoch}, Best Test Accuracy={best_test_perf}, Epoch={best_test_epoch}')
