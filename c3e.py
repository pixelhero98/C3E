import numpy as np
from scipy.optimize import minimize
from typing import Optional, List

class CCCE:
    """
    Estimate propagation layers and their hidden dimensions for a graph neural network with C3E.
    """

    def __init__(self, data, eta, sigma_s: Optional[np.ndarray] = None) -> None:
        """
        :param data: Graph data object with attributes:
                     - x: node feature matrix of shape (N, M)
                     - num_edges: total number of edges
        :param sigma_s: optional per-layer propagation matrix variance scales array
        """
        self.data = data
        self.N, self.M = data.x.shape
        self.d = data.num_edges / self.N
        self.penalty = 1e5
        self.sigma_s = sigma_s
        self.eta = eta

    def dropouts(self, layers: List[float]) -> List[float]:
        """
        Compute dropout probabilities between successive layers.
        """
        w = np.array(layers)
        w = np.insert(w, 0, self.M)
        c = (w[:-1] * w[1:]) / (w[:-1] + w[1:])
        return (1 - c / w[1:]).tolist()

    def rep_compression_ratio(self, layers: List[float], channel_capacity: float) -> float:
        """
        Compute representation compression ratio: network channel capacity divided by geometric mean of hidden dimensions.
        """
        prod = np.prod(layers)
        geo_mean = prod ** (1.0 / len(layers))
        return channel_capacity / geo_mean

    def regularized_feature_dim(self) -> float:
        """
        Regularize feature dimension based on average node degree.
        """
        if self.M <= 0:
            raise ValueError("The number of features (M) must be > 0.")
        return self.M * (self.d ** (1.0 / self.d)) if self.d > 0 else float(self.M)

    def objective(self, w: np.ndarray) -> float:
        """
        Objective: negative channel capacity (to minimize).
        """
        reg_m = self.regularized_feature_dim()
        # penalize widths exceeding reg_m + 1 margin
        if np.any(w > (reg_m + 1)):
            return self.penalty

        term1 = np.log(2 * np.pi * np.e)
        term2 = np.sum(np.log(w[:-1])) + np.log(self.M)
        term3 = np.sum(np.log(self.sigma_s)) if self.sigma_s is not None else np.sum(np.log(1.0 / self.d))
        term4 = len(w) * np.log(self.N)
        return -0.5 * (term1 + term2 + term3 + term4)

    def constraint(self, w: np.ndarray, H: float = 0.0) -> float:
        """
        Inequality constraint: sum of log-terms minus H >= 0.
        """
        terms = []
        for i in range(len(w)):
            if i == 0:
                terms.append(np.log(self.M * w[0] / (self.M + w[0])))
            else:
                num = np.log(w[i-1] * w[i] / (w[i-1] + w[i]))
                terms.append(num / (i + 1))
        # Plain approximation
        # for i in range(len(w)):
        #     if i == 0:
        #         numerator = np.log(self.M * w[0] / (self.M + w[0]))
        #         denom = np.log(2*np.pi*np.e)/np.log(self.N*self.M*self.sigma_s[i]) + i + 1
        #         terms.append(numerator/denom)
        #     else:
        #         numerator = np.log(w[i-1]*w[i]/(w[i-1]+w[i]))
        #         denom = np.log(2*np.pi*np.e)/np.log(self.N*w[i-1]*self.sigma_s[i]) + i + 1
        #         terms.append(numerator/denom)
        return float(np.sum(terms) - H)

    def optimize_weights(
        self,
        H: float,
        verbose: bool = False,
        max_layers: int = 100
    ) -> dict:
        """
        Estimate layer widths under lower bound constraint H, up to max_layers.
        Returns dict with L, weights, entropy, constraint_value.
        """
        for L in range(2, max_layers + 1):
            w0 = np.full(L, 2.0)
            upper = (self.N - 1) * self.M
            bounds = [(2.0, upper)] * L
            cons = {'type': 'ineq', 'fun': lambda w, H=H: self.constraint(w, H)}

            result = minimize(
                fun=self.objective,
                x0=w0,
                method='SLSQP',
                bounds=bounds,
                constraints=[cons]
            )

            if not result.success:
                if verbose:
                    print(f"[L={L}] Optimization failed: {result.message}")
                continue

            weights = result.x
            constraint_val = self.constraint(weights, H)
            entropy = -self.objective(weights)

            if verbose:
                layers_list = weights.tolist()
                print("\n=== Estimation Summary ===")
                print(f"Depth                : {L}")
                print(f"Hidden dimensions    : {layers_list}")
                # Rounded hidden dimensions
                rounded_weights = [int(round(w)) for w in weights]
                print(f"Rounded hidden dims  : {rounded_weights}")
                print(f"Network channel capacity: {entropy:.4f}")
                print(f"Lower bound          : {constraint_val + H:.4f} in [{H:.4f}, {(H / self.eta):.4f}]")
                print(f"Dropout probabilities: {self.dropouts(layers_list)}")
                print(f"Rep. compression     : {self.rep_compression_ratio(layers_list, entropy):.4f}")

            if constraint_val + H > H / self.eta:
                return {
                    'L': L,
                    'weights': weights,
                    'entropy': entropy,
                    'constraint_value': constraint_val
                }

        raise RuntimeError(f"Failed to meet lower bound constraint H={H} within {max_layers} layers.")
