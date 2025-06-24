from Implementations.utility import *
from Model_factory.model import Model


def representation_entropy(sol_path, device, prop_layer, num_class, drop_probs, use_activations, conv_methods, data, nbins):
  
  conv_representation, act_representation = get_model_rep(load_model(Model, sol_path, device, prop_layer, num_class, drop_probs, use_activations, conv_methods), data)
  representation_entropy = show_layer_rep_entropy(conv_representation, act_representation, data, nbins)

  return representation_entropy


def dirichlet_energy(sol_path, device, prop_layer, num_class, drop_probs, use_activations, conv_methods, data, normalized=False):

  conv_representation, act_representation = get_model_rep(load_model(Model, sol_path, device, prop_layer, num_class, drop_probs, use_activations, conv_methods), data)
  dirichlet_energy = show_layer_dirichlet_energy(conv_representation, act_representation, data, normalized)

  return dirichlet_energy
