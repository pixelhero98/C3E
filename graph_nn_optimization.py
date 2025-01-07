import torch
import numpy as np
from scipy.optimize import minimize


class GraphNNOptimization:
    def __init__(self, data):

        self.data = data
        self.N, self.M = data.x.shape
        self.edge_index, self.edge_attr = data.edge_index, data.edge_attr

    def dropouts(self, layers):

        p = []
        for idx in range(len(layers) - 1):
            w0, w1 = layers[idx], layers[idx + 1]
            c = (w0 * w1) / (w0 + w1)
            p.append(1 - (c / w1))

        return p

    def aspect_ratio(self, layers):

        if len(layers) <= 1:
            raise ValueError("Aspect ratio requires at least two layers (excluding the input layer).")
        prod = np.prod(layers)
        depth = len(layers)
        geo_mean = prod ** (1 / depth)

        return depth / geo_mean

    def calculate_average_node_degree(self):

        total_edges = self.data.num_edges
        num_nodes = self.N
        if num_nodes == 0:
            raise ValueError("The graph must contain at least one node.")

        return total_edges / num_nodes

    def regularized_feature_dim(self):

        if self.M == 0:
            raise ValueError("The number of features (M) must be greater than 0.")
        reg = self.calculate_average_node_degree() # Fast approximation of GCN
        # reg = var_propagation, for other models please calculate the variance of delta.
        
        if reg >= 1:
            reg_m = self.M * (reg ** (1/reg))
        else:
            reg_m = self.M

        return reg_m

    def objective(self, x):

        reg_m = self.regularized_feature_dim()
        d = self.calculate_average_node_degree()
        w = x
        psi = 100000

        if any(w_i > reg_m for w_i in w):
            return psi

        return -0.5 * (np.log(2 * np.pi * np.exp(1)) + np.sum(np.log(w[:-1]/d)) + np.log(w[-1]) + np.log(len(w[:-1] * self.N))) # Fast approximation of GCN
        # -0.5 * (np.log(2 * np.pi * np.exp(1)) + np.sum(np.log(w[:-1] * delta)) + np.log(w[-1]) + np.log(len(w[:-1] * self.N))), for other model please calculate delta (variance of propagation matrix Delta)
    def constraint(self, x, H):

        w = x
        terms = [np.log(self.M * w[0] / (self.M + w[0]))]

        for i in range(1, len(w)):
            terms.append((1 / (i + 1)) * np.log(w[i - 1] * w[i] / (w[i - 1] + w[i])))

        return sum(terms) - H

    def optimize_weights(self, H, verbose=False):

        L = 2  # Initialize with at least 2 message-passing layers

        while True:
            w_initial = np.ones(L)  # Initialize widths for these layers
            bounds = [(1, (self.N - 1) * self.M)]  # Bounds for weights

            # Perform the optimization given L
            constraints = [{'type': 'ineq', 'fun': lambda w: self.constraint(w, H)}]
            result = minimize(lambda w: self.objective(w), w_initial, method='SLSQP', bounds=bounds,
                              constraints=constraints)

            if result.success:
                current_constraints = self.constraint(result.x, H)

                if verbose:
                    widths = result.x
                    entropy = -self.objective(result.x)
                    self.print_optimization_results(L, widths, entropy, current_constraints, H)

                if current_constraints > (1/0.49) * H: # eta = 0.49, this can be tuned 
                    break
            else:
                print("Optimization failed for L =", L, ". Please check the constraints and initial conditions.")

            # Increment L and try for next depth
            L += 1

    def print_optimization_results(self, L, weights, current_entropy, current_constraints, H):
        layers = [self.M] + weights.tolist()
        print(f"Number of Propagation Layers: {L}")
        print(f"Weights: {weights.tolist()}")
        print(f"Current Model Entropy: {current_entropy}")
        print(f"Current Constraints/Maximum Graph Entropy: {current_constraints}/{H}")
        print(f"Dropout Probabilities: {self.dropouts(layers)}")
        print(f"Aspect Ratio: {self.aspect_ratio(layers)}")
        print("\n")
