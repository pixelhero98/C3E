import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AAGNN(nn.Module):

    def __init__(self, num_mmp_layer, num_postpro_layer, num_skip_layer):
        super(AAGNN, self).__init__()
        self.num_mmp_layer = num_mmp_layer
        self.num_postpro_layer = num_postpro_layer
        self.num_skip_layer = num_skip_layer

        self.mmp_layer = nn.ModuleList([GCNConv(num_mmp_layer[i], num_mmp_layer[i + 1]) for i in range(len(num_mmp_layer) - 1)])
        self.postpro_layer = nn.ModuleList([nn.Linear(num_postpro_layer[i], num_postpro_layer[i + 1]) for i in range(len(num_postpro_layer) - 1)])
        self.skip_conn_layer = nn.ModuleList([nn.Linear(num_skip_layer[i], num_skip_layer[i + 1]) for i in range(len(num_skip_layer) - 1)])

        self.activation = nn.ModuleList([nn.PReLU() for _ in range(len(num_mmp_layer) - 1)])
        self.activation1 = nn.PReLU()

    def forward(self, x, a, p):

        z = x

        for i in range(len(self.num_mmp_layer) - 1):

            if i == 0:
                z = self.mmp_layer[i](z, a)
                z = self.activation1(z)
                z = F.dropout(z, p[i], training=self.training)
            else:
                residual = self.skip_conn_layer[i - 1](z)
                z = self.mmp_layer[i](z, a) + residual
                z = F.dropout(z, p[i], training=self.training)

        for i in range(len(self.num_postpro_layer) - 1):
            z = self.postpro_layer[i](z)

        return z


def train(model, data, p, optimizer):
    
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, p)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def val(model, data, p):
    
    model.eval()
    pred = model(data.x, data.edge_index, p).argmax(dim=1)
    correct = int((pred[data.val_mask] == data.y[data.val_mask]).sum())
    num_nodes_validation = int(data.val_mask.sum())
    acc = correct / num_nodes_validation

    return acc


@torch.no_grad()
def test(model, data, p):
    
    model.eval()
    pred = model(data.x, data.edge_index, p).argmax(dim=-1)
    correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    num_nodes_test = int(data.test_mask.sum())
    acc = correct / num_nodes_test

    return acc


# For generating the masks for co-purchase graphs
def create_balanced_masks(y, num_classes=15, train_per_class=20, num_val=4700, num_test=13333):

    num_nodes = y.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        class_idx = (y == c).nonzero(as_tuple=False).view(-1)
        class_idx = class_idx[torch.randperm(class_idx.size(0))]

        train_mask[class_idx[:train_per_class]] = True

    remaining_indices = torch.where(~train_mask)[0]
    remaining_indices = remaining_indices[torch.randperm(remaining_indices.size(0))]

    val_mask[remaining_indices[:num_val]] = True
    test_mask[remaining_indices[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask