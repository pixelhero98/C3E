import numpy as np
from scipy.optimize import minimize
from typing import Optional, List

class ChanCapConEst:
    """
    Channel Capacity Constrained Estimator (C3E).

    - Maximizes theoretical channel capacity φ (Eq. 9)
    - Constrains effective capacity φ₀ (Eq. 11) to the window ln n ≤ φ₀ ≤ (1/η) ln n (Eq. 12)
    - Applies Corollary-2 guardrails on the geometric-mean width
    """

    def __init__(
        self,
        data,
        eta: float,
        sigma_s: Optional[Sequence[float] | float] = None,
    ) -> None:
        """
        Args:
            data: object with at least .x (shape: [N, M]).
                  If available, .d (avg degree) or .num_edges is used to estimate degree.
            eta:  regularizer in (0, 1]; controls the φ₀ window's upper bound (1/η) ln n.
            sigma_s: scalar or sequence for σ_{S_l}^2. If None -> fallback 1/d (null approximation).
        """
        if not (0.0 < float(eta) <= 1.0):
            raise ValueError("eta must be in (0, 1].")
        self.eta = float(eta)

        if not hasattr(data, "x"):
            raise ValueError("data.x required (shape [N, M]).")
        self.N, self.M = map(int, data.x.shape)

        # avg degree (fallbacks safely to 1.0)
        d_from_attr = float(getattr(data, "d", 0.0) or 0.0)
        d_from_edges = 0.0
        if hasattr(data, "num_edges"):
            try:
                d_from_edges = 2.0 * float(data.num_edges) / float(self.N)
            except Exception:
                d_from_edges = 0.0
        self.d = float(d_from_attr if d_from_attr > 0 else (d_from_edges if d_from_edges > 0 else 1.0))

        self.sigma_s = sigma_s  # may be None, scalar, or sequence
        self.penalty = 1e6

    def regularized_feature_dim(self) -> float:
        """
        Regularize feature dimension to enable controlled and stable solutions.
        """
        if self.M <= 0:
            raise ValueError("The number of features (M) must be > 0.")
        elif self.sigma_s is not None:
            # treat input as scalar (if array, use first value)
            ss = float(np.atleast_1d(np.asarray(self.sigma_s, dtype=float))[0])
            return self.M * ((1.0 / (ss * self.N)) ** (ss * self.N))
        else:
            return self.M * (self.d ** (1.0 / self.d)) if self.d > 0 else float(self.M)

    # -------------------------
    # Internals
    # -------------------------
    def _sigma2(self, L: int) -> np.ndarray:
        """
        Per-layer σ_{S_l}^2 as a length-L array.
        Uses fallback σ^2 = 1/d when sigma_s is None.
        """
        if self.sigma_s is None:
            val = 1.0 / self.d if self.d > 0 else 1.0
            return np.full(L, val, dtype=float)
        arr = np.atleast_1d(np.asarray(self.sigma_s, dtype=float))
        if arr.size == 1:
            return np.full(L, float(arr[0]), dtype=float)
        if arr.size != L:
            raise ValueError(f"sigma_s length ({arr.size}) must be 1 or equal to L={L}.")
        return arr

    @staticmethod
    def _geom_mean_width(w: np.ndarray) -> float:
        return float(np.exp(np.mean(np.log(w))))

    def _K_wstar_terms(self, L: int) -> Tuple[float, float]:
        """
        Returns:
            K       = E_l [ ln( n * σ_{S_l}^2 ) ]
            e^{-K}  = exp(-K)  (used in guardrail ln w̄ > -K)
        """
        sigma2 = self._sigma2(L)
        K = float(np.mean(np.log(self.N) + np.log(sigma2)))
        return K, float(np.exp(-K))

    # -------------------------
    # Core equations
    # -------------------------
    def theoretical_capacity(self, w: np.ndarray) -> float:
        """
        φ = 0.5 * [ ln(2πe) + Σ_{l=1..L} ln( n * w_{l-1} * σ_{S_l}^2 ) ], with w0 = M.  (Eq. 9)
        """
        L = w.size
        sigma2 = self._sigma2(L)
        prev = np.concatenate(([self.M], w[:-1]))  # w_{l-1}
        terms = np.log(self.N) + np.log(prev) + np.log(sigma2)
        return 0.5 * (np.log(2.0 * np.pi * np.e) + np.sum(terms))

    def effective_capacity(self, w: np.ndarray) -> float:
        """
        If σ_S^2 is scalar (or None → single value), use 1/l scaling in front of log-HM.
        Otherwise use the exact Eq. (11) with the ratio DEN / NUM.
        """
        L = w.size
        prev = np.concatenate(([self.M], w[:-1]))              # w_{l-1}
        hm_logs = np.log((prev * w) / (prev + w))              # ln((w_{l-1} w_l)/(w_{l-1}+w_l))
    
        # Case 1: all-layer-same σ^2  →  scale by 1/l
        if (self.sigma_s is None or np.isscalar(self.sigma_s) or
            (np.ndim(self.sigma_s) == 1 and np.atleast_1d(self.sigma_s).size == 1)):
            scale = 1.0 / np.arange(1, L + 1, dtype=float)
            return float(np.sum(hm_logs * scale))
    
        # Case 2: layer-varying σ^2  →  exact Eq. (11)
        sigma2 = self._sigma2(L)
        den_logs = np.log(self.N) + np.log(prev) + np.log(sigma2)   # ln(n w_{l-1} σ_{S_l}^2)  (DEN)
        num_caps = np.log(2.0 * np.pi * np.e) + np.cumsum(den_logs) # ln(2πe) + Σ_{o≤l} ln(n w_{o-1} σ_{S_o}^2) (NUM)
    
        with np.errstate(divide="ignore", invalid="ignore"):
            per = hm_logs * (den_logs / num_caps)                   # <-- DEN / NUM (correct)
            per[~np.isfinite(per)] = -np.inf
    
        return float(np.sum(per))

    # -------------------------
    # Constraints (Eq. 12 + Corollary 2)
    # -------------------------
    def _ineq_lower_cap(self, w: np.ndarray, H: float) -> float:
        # φ₀ - H ≥ 0
        return self.effective_capacity(w) - H

    def _ineq_upper_cap(self, w: np.ndarray, H: float) -> float:
        # (H/η) - φ₀ ≥ 0
        return (H / self.eta) - self.effective_capacity(w)

    def _ineq_cor_logw(self, w: np.ndarray) -> float:
        # ln w̄ > -K  ⇔ w̄ > e^{-K}
        _, e_negK = self._K_wstar_terms(w.size)
        return self._geom_mean_width(w) - e_negK

    def _ineq_cor_wstar(self, w: np.ndarray) -> float:
        # w̄ > w* ≈ e^{1-K}
        K, _ = self._K_wstar_terms(w.size)
        w_star = float(np.exp(1.0 - K))
        return self._geom_mean_width(w) - w_star
        
    # -------------------------
    # Objective and utilities
    # -------------------------
    def objective(self, w: np.ndarray) -> float:
        """
        Minimize negative theoretical capacity (Eq. 9).
        No extra penalties/knobs; feasibility handled by constraints and bounds.
        """
        phi = self.theoretical_capacity(w)
        if np.any(w <= 0) or not np.isfinite(phi):
            return self.penalty
        return -phi

    def dropouts(self, layers: List[float]) -> List[float]:
        """
        Harmonic-mean bottleneck 'dropout' scores between successive layers.
        """
        w = np.array(layers, dtype=float)
        w_prev = np.insert(w, 0, self.M)
        c = (w_prev[:-1] * w_prev[1:]) / (w_prev[:-1] + w_prev[1:])
        return (1.0 - c / w_prev[1:]).tolist()

    def rep_compression_ratio(self, layers: List[float], channel_capacity: float) -> float:
        """
        θ (Eq. 10): φ / geometric-mean(widths).
        """
        w = np.array(layers, dtype=float)
        return float(channel_capacity / self._geom_mean_width(w))

    # -------------------------
    # Solver
    # -------------------------
    def optimize_weights(self, H: float, verbose: bool = False, max_layers: int = 100) -> list:
        """
        Constrained search over L = 2..max_layers.
        Bounds:
            2 ≤ w_l ≤ regularized_feature_dim() + 1   (your empirical cap)
        Constraints:
            ln n ≤ φ₀ ≤ (1/η) ln n,  ln w̄ > -K,  w̄ > e^{1-K}.
        Returns:
            [rounded_widths_per_L, dropouts_per_L, phi_per_L]
        """
        all_rounded, all_dropouts, all_phi = [], [], []

        # hard per-layer upper bound from your empirical rule (no new knobs)
        reg_m = float(self.regularized_feature_dim())
        ub = max(2.0, reg_m + 1.0)

        for L in range(2, max_layers + 1):
            # bounds & init
            bounds = [(2.0, ub)] * L
            w0_val = min(max(2.0, float(self.M)), ub - 1e-6)
            w0 = np.full(L, w0_val, dtype=float)

            cons = [
                {"type": "ineq", "fun": lambda w, H=H: self._ineq_lower_cap(w, H)},
                {"type": "ineq", "fun": lambda w, H=H: self._ineq_upper_cap(w, H)},
                {"type": "ineq", "fun": self._ineq_cor_logw},
                {"type": "ineq", "fun": self._ineq_cor_wstar},
            ]

            res = minimize(
                fun=self.objective,
                x0=w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 200},
            )

            if not res.success:
                if verbose:
                    print(f"[L={L}] infeasible: {res.message}")
                continue

            w = res.x
            phi = self.theoretical_capacity(w)
            rounded = [int(round(v)) for v in w]
            drops = self.dropouts(w.tolist())

            if verbose:
                phi0 = self.effective_capacity(w)
                print("\n=== C3E (feasible) ===")
                print(f"L={L}\nwidths: {w.tolist()} (rounded {rounded})")
                print(f"φ (Eq.9): {phi:.6f} | φ₀ (Eq.11): {phi0:.6f} in [{H:.6f}, {(H/self.eta):.6f}]")
                print(f"dropouts: {drops}")

            all_rounded.append(rounded)
            all_dropouts.append(drops)
            all_phi.append(phi)

        if not all_rounded:
            raise RuntimeError(f"No feasible (L, w) satisfied constraints in 2..{max_layers}.")
        return [all_rounded, all_dropouts, all_phi]
