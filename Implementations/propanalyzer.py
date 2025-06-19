import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, identity
from torch_geometric.data import Data
from torch_geometric.transforms import GDC
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix


def scaled_laplacian(A_csr: csr_matrix) -> csr_matrix:
    """Compute scaled Laplacian."""
    row_sums = np.array(A_csr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    inv_sqrt = 1.0 / np.sqrt(row_sums)
    D_inv_sqrt = diags(inv_sqrt)
    L = identity(A_csr.shape[0], format='csr') - D_inv_sqrt @ A_csr @ D_inv_sqrt
    return 2 * L - identity(A_csr.shape[0], format='csr')


class PropagationVarianceAnalyzer:
    """
    Computes variance of the propagation matrix for various GNN-style
    propagation schemes. Supported methods: 'gcn', 'appnp', 'gdc', 'sgc',
    'chebnetii', 'gprgnn', 'jacobiconv', 's2gc', or 'all'.
    """
    SUPPORTED = {
        'gcn', 'appnp', 'gdc', 'sgc',
        'chebnetii', 'gprgnn', 'jacobiconv',
        's2gc', 'all'
    }

    def __init__(
        self,
        data,
        method: str = 'all',
        # APPNP
        appnp_k: int = 10,
        appnp_alpha: float = 0.1,
        # GDC diffusion
        gdc_method: str = 'ppr',
        alpha: float = 0.05,
        heat_t: float = 1.0,
        # GDC sparsification
        gdc_spars_method: str = 'topk',
        gdc_avg_degree: int = 64,
        gdc_threshold_eps: float = 1e-4,
        gdc_exact: bool = False,
        # SGC
        sgc_k: int = 2,
        # ChebNet II
        cheb_k: int = 10,
        cheb_theta: np.ndarray = None,
        # GPRGNN
        gpr_k: int = 10,
        gpr_theta: np.ndarray = None,
        # JacobiConv
        jacobi_iters: int = 10,
        jacobi_alpha: float = 0.5,
        # S2GC
        s2gc_k: int = 1,
        s2gc_alpha: float = 0.5,
    ):
        self.method = method.lower()
        if self.method not in self.SUPPORTED:
            raise ValueError(f"Unsupported method '{method}'. Choose from {self.SUPPORTED}.")

        # Build adjacency CSR
        if isinstance(data, Data):
            edge_weight = getattr(data, 'edge_weight', torch.ones(data.edge_index.size(1)))
            self.A = to_scipy_sparse_matrix(
                data.edge_index, edge_weight, num_nodes=data.num_nodes
            ).tocsr()
            self.data = data
        elif isinstance(data, csr_matrix):
            self.A = data
            # Reconstruct a minimal PyG Data for methods that need edge_index
            edge_index, edge_weight = from_scipy_sparse_matrix(self.A)
            self.data = Data(edge_index=edge_index, edge_weight=edge_weight)
        else:
            raise TypeError("`data` must be a PyG Data or scipy.sparse.csr_matrix")

        # Store parameters
        self.appnp_k          = appnp_k
        self.appnp_alpha      = appnp_alpha
        self.alpha            = alpha
        self.heat_t           = heat_t
        self.gdc_spars_method = gdc_spars_method.lower()
        self.gdc_avg_degree   = gdc_avg_degree
        self.gdc_threshold_eps= gdc_threshold_eps
        self.gdc_exact        = gdc_exact
        self.sgc_k            = sgc_k
        self.cheb_k           = cheb_k
        self.cheb_theta       = (
            cheb_theta
            if cheb_theta is not None
            else np.ones(self.cheb_k + 1) / (self.cheb_k + 1)
        )
        self.gpr_k            = gpr_k
        self.gpr_theta        = (
            gpr_theta
            if gpr_theta is not None
            else np.ones(self.gpr_k + 1) / (self.gpr_k + 1)
        )
        self.jacobi_iters     = jacobi_iters
        self.jacobi_alpha     = jacobi_alpha
        self.s2gc_k           = s2gc_k
        self.s2gc_alpha       = s2gc_alpha

        if self.gdc_spars_method not in {'topk', 'threshold'}:
            raise ValueError("gdc_spars_method must be 'topk' or 'threshold'")
        self.gdc_method = gdc_method

        # Cache for propagation matrices
        self._cache = {}

    def _prop_gcn(self) -> csr_matrix:
        """Symmetric normalization propagation with self-loops (GCN)."""
        edge_index, edge_weight = gcn_norm(
            self.data.edge_index,
            getattr(self.data, 'edge_weight', None),
            num_nodes=self.A.shape[0],
            improved=False,
            add_self_loops=True
        )
        return to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=self.A.shape[0])

    def _prop_appnp(self) -> csr_matrix:
        """Approximate personalized propagation of neural predictions (APPNP)."""
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        P = diags(1.0 / row_sums) @ self.A
        M = P.multiply(1 - self.appnp_alpha)
        N = self.A.shape[0]
        S = identity(N, format='csr') * self.appnp_alpha
        M_power = identity(N, format='csr')
        for _ in range(1, self.appnp_k):
            M_power = M @ M_power
            S += self.appnp_alpha * M_power
        S += M @ M_power
        return S

    def _prop_gdc(self) -> csr_matrix:
        """Graph diffusion convolution (GDC) with PPR or heat diffusion."""
        N = self.A.shape[0]
        # diffusion kwargs
        diff_kwargs = {'method': 'heat', 't': self.heat_t} if self.gdc_method == 'heat' else {'method': 'ppr', 'alpha': self.alpha}
        # sparsification kwargs
        if self.gdc_spars_method == 'topk':
            spars_kwargs = {'method': 'topk', 'k': self.gdc_avg_degree, 'dim': 0}
        else:
            spars_kwargs = {'method': 'threshold', 'eps': self.gdc_threshold_eps}

        transform = GDC(
            self_loop_weight=1.0,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=diff_kwargs,
            sparsification_kwargs=spars_kwargs
        )
        data_gdc = transform(self.data)
        return to_scipy_sparse_matrix(
            data_gdc.edge_index,
            data_gdc.edge_attr,
            num_nodes=N
        ).tocsr()

    def _prop_sgc(self) -> csr_matrix:
        """Simple graph convolution (SGC) by K-step power of normalized adjacency."""
        N = self.A.shape[0]
        A_hat = self.A + identity(N, format='csr')
        row_sums = np.array(A_hat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        norm_A = diags(inv_sqrt) @ A_hat @ diags(inv_sqrt)
        M = norm_A
        for _ in range(1, self.sgc_k):
            M = M @ norm_A
        return M

    def _prop_chebnetii(self) -> csr_matrix:
        """Chebyshev spectral graph convolution (ChebNet II)."""
        L_tilde = scaled_laplacian(self.A)
        N = self.A.shape[0]
        j = np.arange(self.cheb_k + 1)
        xj = np.cos((2*j + 1) * np.pi / (2*(self.cheb_k + 1)))
        T_prev, T_curr = identity(N, format='csr'), L_tilde
        S = csr_matrix((N, N))
        for k in range(self.cheb_k + 1):
            T_k = T_prev if k == 0 else (T_curr if k == 1 else 2 * L_tilde @ T_curr - T_prev)
            if k > 1:
                T_prev, T_curr = T_curr, T_k
            Tk_xj = np.cos(k * np.arccos(xj))
            w_k = 2.0 / (self.cheb_k + 1) * (self.cheb_theta * Tk_xj).sum()
            S += w_k * T_k
        return S

    def _prop_gprgnn(self) -> csr_matrix:
        """Generalized PageRank neural network (GPRGNN)."""
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        P = diags(1.0 / row_sums) @ self.A
        N = self.A.shape[0]
        S = csr_matrix((N, N))
        P_power = identity(N, format='csr')
        for k in range(self.gpr_k + 1):
            S += self.gpr_theta[k] * P_power
            P_power = P @ P_power
        return S

    def _prop_jacobiconv(self) -> csr_matrix:
        """Jacobi-based propagation (JacobiConv)."""
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D_inv = diags(1.0 / row_sums)
        M = self.jacobi_alpha * (D_inv @ self.A)
        N = self.A.shape[0]
        S = identity(N, format='csr')
        M_power = identity(N, format='csr')
        for _ in range(self.jacobi_iters):
            M_power = M @ M_power
            S += M_power
        return S

    def _prop_s2gc(self) -> csr_matrix:
        """Second-order graph convolution (S2GC)."""
        N = self.A.shape[0]
        A_hat = self.A + identity(N, format='csr')
        row_sums = np.array(A_hat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        norm_A = diags(inv_sqrt) @ A_hat @ diags(inv_sqrt)
        S = self.s2gc_alpha * identity(N, format='csr')
        M_power = identity(N, format='csr')
        for _ in range(self.s2gc_k):
            M_power = norm_A @ M_power
            S += (1.0 - self.s2gc_alpha) / self.s2gc_k * M_power
        return S

    def compute_variance(self, method: str = None) -> float:
        """Compute variance of the propagation matrix without densification."""
        m = (method or self.method).lower()
        if m == 'all':
            raise ValueError("Use compute_all() to get all methods")
        if m not in self.SUPPORTED:
            raise ValueError(f"Unknown method '{m}'")
        if m not in self._cache:
            self._cache[m] = getattr(self, f"_prop_{m}")()
        S = self._cache[m]
        n = S.shape[0]
        total = float(S.sum())
        sum_sq = float((S.data ** 2).sum())
        N2 = n * n
        mean = total / N2
        mean_sq = sum_sq / N2
        return mean_sq - mean ** 2

    def compute_all(self) -> dict:
        """Compute variances for all supported propagation methods."""
        return {m: self.compute_variance(m) for m in self.SUPPORTED if m != 'all'}
