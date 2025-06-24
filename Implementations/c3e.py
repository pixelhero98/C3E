import numpy as np
from scipy.optimize import minimize
from typing import Optional, List

class ChanCapConEst:
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
        Regularize feature dimension to enable controlled and stable solutions.
        """
        if self.M <= 0:
            raise ValueError("The number of features (M) must be > 0.")
        elif self.sigma_s is not None:
            # use self.N in place of undefined 'n'
            return self.M * ((1.0 / (self.sigma_s * self.N)) ** (self.sigma_s * self.N))
        else:
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
        term3 = (np.sum(np.log(self.sigma_s))
                 if self.sigma_s is not None
                 else np.sum(np.log(1.0 / self.d)))
        term4 = len(w) * np.log(self.N)
        term5 = np.log(w[-1])
        return -0.5 * (term1 + term2 + term3 + term4 + term5)

    def constraint(self, w: np.ndarray, H: float = 0.0) -> float:
        """
        Inequality constraint: sum of log-terms minus H >= 0.
        """
        terms = []
        i = np.arange(len(w)) + 1
        if self.sigma_s is None:
            numerators = np.empty_like(w)
            numerators[0] = np.log(self.M * w[0] / (self.M + w[0]))
            numerators[1:] = np.log(w[:-1] * w[1:] / (w[:-1] + w[1:]))
            terms = numerators / i
        else:
            log_constants = np.log(2 * np.pi * np.e)
            tmp = np.empty_like(w, dtype=float)
            tmp[0] = log_constants / np.log(self.N * self.M * self.sigma_s) + 1
            tmp[1:] = log_constants / np.log(self.N * w[:-1] * self.sigma_s) + i[1:]
            numerators = np.empty_like(w)
            numerators[0] = np.log(self.M * w[0] / (self.M + w[0]))
            numerators[1:] = np.log(w[:-1] * w[1:] / (w[:-1] + w[1:]))
            terms = numerators / tmp
        return float(terms.sum() - H)
     
    def optimize_weights(
        self,
        H: float,
        verbose: bool = False,
        max_layers: int = 100
    ) -> list:
        """
        Estimate layer widths under lower bound constraint H, up to max_layers.
        Returns a list where the first element is the list of rounded hidden dimensions
        for each L, and the second element is the list of dropout probabilities for each L.
        """
        all_rounded = []
        all_dropouts = []
        all_channel_capacity = []
        
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

            # compute rounded hidden dimensions and dropout probabilities
            rounded_weights = [int(round(wi)) for wi in weights]
            dropout_probs = self.dropouts(weights.tolist())

            all_rounded.append(rounded_weights)
            all_dropouts.append(dropout_probs)
            all_channel_capacity.append(entropy)

            if verbose:
                print("\n=== Estimation Summary ===")
                print(f"Depth                : {L}")
                print(f"Hidden dimensions    : {weights.tolist()}")
                print(f"Rounded hidden dims  : {rounded_weights}")
                print(f"Network channel capacity: {entropy:.4f}")
                print(f"Lower bound          : {constraint_val + H:.4f} in [{H:.4f}, {(H / self.eta):.4f}]")
                print(f"Dropout probabilities: {dropout_probs}")
                print(f"Rep. compression     : {self.rep_compression_ratio(weights.tolist(), entropy):.4f}")

            # return all results up to the first L that meets the constraint
            if constraint_val + H > H / self.eta:
                return [all_rounded, all_dropouts, all_channel_capacity]

        raise RuntimeError(f"Failed to meet lower bound constraint H={H} within {max_layers} layers.")
