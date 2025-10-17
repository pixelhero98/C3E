"""C3E channel capacity constrained estimator implementation."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

__all__ = ["ChanCapConEst"]


TINY = 1e-12

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
        self.penalty = 1e6
        self.sigma_s = sigma_s
        self.eta = eta

    def dropouts(self, layers: Sequence[float]) -> List[float]:
        """Compute dropout probabilities between successive layers."""
        widths = np.array(layers, dtype=float)
        widths = np.insert(widths, 0, self.M)
        harmonic = (widths[:-1] * widths[1:]) / (widths[:-1] + widths[1:])
        return (1.0 - harmonic / widths[1:]).tolist()

    def rep_compression_ratio(self, layers: Sequence[float], channel_capacity: float) -> float:
        """Compute channel capacity divided by the geometric mean of hidden dimensions."""
        prod = np.prod(np.asarray(layers, dtype=float))
        geo_mean = prod ** (1.0 / len(layers))
        return channel_capacity / geo_mean

    def regularized_feature_dim(self) -> float:
        """
        Regularize feature dimension to enable controlled and stable solutions.
        """
        if self.M <= 0:
            raise ValueError("The number of features (M) must be > 0.")
        elif self.sigma_s is not None:
            return self.M * ((1.0 / (self.sigma_s * self.N)) ** (self.sigma_s * self.N))
        else:
            return self.M * (self.d ** (1.0 / self.d)) if self.d > 0 else float(self.M)

    def _layer_variances(self, length: int) -> np.ndarray:
        """Return per-layer variance values for the given network depth."""
        if self.sigma_s is None:
            d_safe = max(self.d, TINY)
            return np.full(length, 1.0 / d_safe, dtype=float)

        sigma = np.asarray(self.sigma_s, dtype=float).ravel()
        if sigma.size == 1:
            return np.full(length, float(sigma[0]), dtype=float)
        return sigma[:length]

    def _violates_strict_guards(self, w: np.ndarray) -> bool:
        """
        Return True if any strict guard is violated:
          (i) per-layer width upper bound   w_ell <= reg_m + 1
         (ii) ln(w̄) > -K
        (iii) w̄   > exp(1 - K)
        where w̄ is the geometric mean of widths {w_1..w_L} and
              K = E_ell[ ln(n * sigma_S_ell^2) ].
        """
        w = np.asarray(w, dtype=float).ravel()
        L = len(w)
        reg_m = self.regularized_feature_dim()
    
        # (i) strict per-layer cap
        if np.any(w > (reg_m + 1)):
            return True
    
        # K and w̄
        # per-layer variances σ^2
        sigma2 = self._layer_variances(L)

        # K = ln n + mean(ln σ^2_ell)
        K = np.log(self.N + TINY) + np.mean(np.log(sigma2 + TINY))

        # geometric mean of widths
        wbar = np.exp(np.mean(np.log(w + TINY)))

        # (ii) and (iii) guardrails
        if (np.log(wbar) <= -K) or (wbar <= np.exp(1.0 - K)):
            return True

        return False


    def objective(self, w: np.ndarray) -> float:
        # strict guards (OR of all three): if any trip, penalize
        if self._violates_strict_guards(w):
            return self.penalty

        # ---- existing φ(w) code remains the same below ----
        w = np.asarray(w, dtype=float)
        w_ext = np.concatenate(([self.M], w))
        L = len(w)

        sigma2 = self._layer_variances(L)
        per_layer = (
            np.log(self.N + TINY)
            + np.log(w_ext[:-1] + TINY)
            + np.log(sigma2 + TINY)
        )
        phi = 0.5 * (np.log(2.0 * np.pi * np.e) + per_layer.sum())
        return -phi


    def constraint(self, w: np.ndarray, H: float = 0.0) -> float:
        """
        Inequality constraint: φ₀(w) - H >= 0, where
        - If σ_s is constant/omitted:
            φ₀ = Σ_{ℓ=1..L} (1/ℓ) * ln( HM(w_{ℓ-1}, w_ℓ) )
        - If σ_s varies by layer:
            φ₀ = Σ_{ℓ=1..L} ln( HM(w_{ℓ-1}, w_ℓ) ) * [ ln(n w_{ℓ-1} σ_{S_ℓ}^2) / ( ln(2πe) + Σ_{o=1..ℓ} ln(n w_{o-1} σ_{S_o}^2) ) ]
        with HM(a,b) = ab/(a+b) and w0 = M.
        """
        w = np.asarray(w, dtype=float)
        w_ext = np.concatenate(([self.M], w))
        L = len(w)

        # log HM terms h_ℓ
        h = np.log(
            (w_ext[:-1] * w_ext[1:] + TINY)
            / (w_ext[:-1] + w_ext[1:] + TINY)
        )

        # resolve σ^2 per layer
        if self.sigma_s is None:
            sigma2 = self._layer_variances(L)
            # constant-σ case: weights = 1/ℓ
            idx = np.arange(1, L + 1, dtype=float)
            phi0 = (h / idx).sum()
        else:
            sigma2 = self._layer_variances(L)
            if np.allclose(sigma2, sigma2[0]):
                # treat scalar-like arrays as constant-σ
                idx = np.arange(1, L + 1, dtype=float)
                phi0 = (h / idx).sum()
            else:
                # varying-σ case: exact DEN/NUM weights
                den = (
                    np.log(self.N + TINY)
                    + np.log(w_ext[:-1] + TINY)
                    + np.log(sigma2 + TINY)
                )
                num_prefix = np.cumsum(den) + np.log(2.0 * np.pi * np.e)
                weights = den / (num_prefix + TINY)
                phi0 = np.sum(h * weights)

        return float(phi0 - H)

    def optimize_weights(
        self,
        H: float,
        verbose: bool = False,
        max_layers: int = 100
    ) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """
        Estimate layer widths under lower bound constraint H, up to max_layers.
        Returns a list where the first element is the list of rounded hidden dimensions
        for each L, and the second element is the list of dropout probabilities for each L.
        """
        all_rounded: List[List[int]] = []
        all_dropouts: List[List[float]] = []
        all_channel_capacity: List[float] = []
        
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
                return (all_rounded, all_dropouts, all_channel_capacity)

        raise RuntimeError(f"Failed to meet lower bound constraint H={H} within {max_layers} layers.")
