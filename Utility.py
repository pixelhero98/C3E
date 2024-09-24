from gcn_box import *
from gat_box import *
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from graph_nn_optimization import *
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def get_model_rep(loaded_model, data, p):
  
    ops = []
    layers = [loaded_model.activation1] + [l for l in loaded_model.mmp_layer if l is not loaded_model.mmp_layer[0]] + [loaded_model.postpro_layer[0]]
    hooks = []
    try:
        for layer in layers:
            hooks.append(layer.register_forward_hook(lambda module, input, output: ops.append(output)))
        loaded_model(data.x, data.edge_index, p)
    finally:
        for hook in hooks:
            hook.remove()
    return ops


def rep_entropy(rep):

    rep = rep.view(-1)
    p = F.softmax(rep, dim=0)
    uniq, occs = torch.unique(p, return_counts=True)
    p = uniq * occs
  
    return -torch.sum(p * torch.log(p)).item()


def show_layer_rep_entropy(ops, data):
  
    entropy_dict = {}
    entropy_dict[0] = rep_entropy(data.x)

    for idx, i in enumerate(ops, start=1):
        entropy_dict[idx] = rep_entropy(i)
    print(entropy_dict)
    print('===============================================================')


def compute_laplacian(edge_index, num_nodes):
    # Compute the degree matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # edge_index[0] contains source nodes, edge_index[1] contains target nodes
    src_nodes = edge_index[0]
    tgt_nodes = edge_index[1]

    # Set entries of adj_matrix to 1 where there are edges
    adj_matrix[src_nodes, tgt_nodes] = 1
    degree_matrix = torch.diag(adj_matrix.sum(dim=1))

    # Compute the Laplacian
    laplacian_matrix = degree_matrix - adj_matrix
    return laplacian_matrix


def compute_dirichlet_energy(node_features, laplacian_matrix):
    energy = torch.mm(node_features.t(), torch.mm(laplacian_matrix.to(device), node_features))
    return torch.trace(energy).item()  # Return as a Python float


def show_layer_dirichlet_energy(model_rep, data):
    dirichlet_energy_dict = {}
    # Compute the initial Dirichlet energy for the input features
    initial_x = compute_dirichlet_energy(data.x, compute_laplacian(data.edge_index, data.x.shape[0]))

    # Set the first value to 1.0 representing the normalized initial Dirichlet energy
    dirichlet_energy_dict[0] = 1.0

    # Calculate and normalize Dirichlet energy for each representation in model_rep
    for idx, rep in enumerate(model_rep, start=1):  # start index at 1 for subsequent layers
        dirichlet_energy_dict[idx] = compute_dirichlet_energy(rep, compute_laplacian(data.edge_index,
                                                                                     data.x.shape[0])) / initial_x

    print(dirichlet_energy_dict)
    print('===============================================================')



# Functions for statistics of Node representation

def vector_variance(rep):
    return torch.var(rep, dim=0, unbiased=True)

def vector_mean(rep):
    return torch.mean(rep, dim=0)
  
def fla(rep):
  blank = []
  for i in rep:
    blank.append(vector_mean(i.view(-1)).cpu().detach())

  return blank

# Functions for statistics of Node vector
def node_fla(rep):

  blank = []
  for i in rep:
    blank.append(vector_mean(i).cpu().detach())

  return blank
