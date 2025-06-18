import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, identity
from torch_geometric.data import Data
from torch_geometric.transforms import GDC
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix


def scaled_laplacian(A_csr: csr_matrix) -> csr_matrix:
    """Compute scaled Laplacian \tilde{L} = 2L/\lambda_max - I using lambda_max≈2"""
    row_sums = np.array(A_csr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    inv_sqrt = 1.0 / np.sqrt(row_sums)
    D_inv_sqrt = diags(inv_sqrt)
    L = identity(A_csr.shape[0], format='csr') - D_inv_sqrt.dot(A_csr).dot(D_inv_sqrt)
    return 2 * L - identity(A_csr.shape[0], format='csr')


class PropagationVarianceAnalyzer:
    """
    Analyzer for computing variance of propagation matrix for a chosen method
    or all methods. Supported methods: 'gcn', 'appnp', 'gdc', 'sgc',
    'chebnetii', 'gprgnn', 'jacobiconv', 's2gc', 'all'.

    Accepts PyG Data or scipy.sparse.csr_matrix. Defaults align with original works.
    GDC supports 'ppr' or 'heat' diffusion via gdc_method parameter.
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
        # GDC (PPR) teleport probability
        alpha: float = 0.05,
        # APPNP parameters
        appnp_k: int = 10,
        appnp_alpha: float = 0.1,
        # SGC K
        sgc_k: int = 2,
        # ChebNetII
        cheb_k: int = 5,
        cheb_theta: np.ndarray = None,
        # GPRGNN
        gpr_k: int = 10,
        gpr_theta: np.ndarray = None,
        # JacobiConv
        jacobi_iters: int = 10,
        jacobi_alpha: float = 0.5,
        # S2GC (SSGConv) parameters
        s2gc_k: int = 1,
        s2gc_alpha: float = 0.5,
        # GDC specific
        gdc_method: str = 'ppr',  # 'ppr' or 'heat'
        heat_t: float = 1.0,
    ):
        self.method = method.lower()
        if self.method not in self.SUPPORTED:
            raise ValueError(f"Unsupported method '{method}'. Choose from {self.SUPPORTED}.")

        # Build adjacency CSR
        if isinstance(data, Data):
            edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else torch.ones(data.edge_index.size(1))
            self.A = to_scipy_sparse_matrix(
                data.edge_index, edge_weight, num_nodes=data.num_nodes
            ).tocsr()
        elif isinstance(data, csr_matrix):
            self.A = data
            data = None
        else:
            raise TypeError("data must be a PyG Data or scipy.sparse.csr_matrix")
        self.data = data

        # Store parameters
        self.alpha        = alpha
        self.appnp_k      = appnp_k
        self.appnp_alpha  = appnp_alpha
        self.sgc_k        = sgc_k
        self.cheb_k       = cheb_k
        self.cheb_theta   = cheb_theta if cheb_theta is not None else np.ones(self.cheb_k + 1) / (self.cheb_k + 1)
        self.gpr_k        = gpr_k
        self.gpr_theta    = gpr_theta if gpr_theta is not None else np.ones(self.gpr_k + 1) / (self.gpr_k + 1)
        self.jacobi_iters = jacobi_iters
        self.jacobi_alpha = jacobi_alpha
        # S2GC parameters
        self.s2gc_k       = s2gc_k
        self.s2gc_alpha   = s2gc_alpha
        # GDC-specific
        self.gdc_method = gdc_method.lower()
        if self.gdc_method not in {'ppr', 'heat'}:
            raise ValueError("gdc_method must be 'ppr' or 'heat'")
        self.heat_t = heat_t

    def _prop_gcn(self) -> csr_matrix:
        if self.data is None:
            raise ValueError("GCN requires PyG Data input with edge_index")
        edge_index, edge_weight = gcn_norm(
            self.data.edge_index,
            self.data.edge_weight if hasattr(self.data, 'edge_weight') else None,
            num_nodes=self.A.shape[0],
            improved=False,
            add_self_loops=True
        )
        return to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=self.A.shape[0])

    def _prop_appnp(self) -> csr_matrix:
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        P = diags(1.0 / row_sums).dot(self.A)
        M = P.multiply(1 - self.appnp_alpha)
        N = self.A.shape[0]
        S = identity(N, format='csr') * self.appnp_alpha
        M_power = identity(N, format='csr')
        for _ in range(1, self.appnp_k):
            M_power = M.dot(M_power)
            S += self.appnp_alpha * M_power
        S += M.dot(M_power)
        return csr_matrix(S)

    def _prop_gdc(self) -> csr_matrix:
        if self.data is None:
            raise ValueError("GDC requires PyG Data input with edge_index")
        if self.gdc_method == 'ppr':
            diff_kwargs = {'method': 'ppr', 'alpha': self.alpha}
        else:
            diff_kwargs = {'method': 'heat', 't': self.heat_t}
        transform = GDC(
            self_loop_weight=1.0,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=diff_kwargs
        )
        data_gdc = transform(self.data)
        return to_scipy_sparse_matrix(
            data_gdc.edge_index, data_gdc.edge_weight, num_nodes=self.A.shape[0]
        )

    def _prop_sgc(self) -> csr_matrix:
        N = self.A.shape[0]
        A_hat = self.A + identity(N, format='csr')
        row_sums = np.array(A_hat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        norm_A = diags(inv_sqrt).dot(A_hat).dot(diags(inv_sqrt))
        M = norm_A
        for _ in range(1, self.sgc_k):
            M = M.dot(norm_A)
        return csr_matrix(M)

    def _prop_chebnetii(self) -> csr_matrix:
        L_tilde = scaled_laplacian(self.A)
        N = self.A.shape[0]
        j = np.arange(self.cheb_k + 1)
        xj = np.cos((2 * j + 1) * np.pi / (2 * (self.cheb_k + 1)))
        T_prev = identity(N, format='csr')
        T_curr = L_tilde
        S = csr_matrix((N, N))
        for k in range(self.cheb_k + 1):
            if k == 0:
                T_k = T_prev
            elif k == 1:
                T_k = T_curr
            else:
                T_k = 2 * L_tilde.dot(T_curr) - T_prev
                T_prev, T_curr = T_curr, T_k
            Tk_xj = np.cos(k * np.arccos(xj))
            w_k = 2.0 / (self.cheb_k + 1) * (self.cheb_theta * Tk_xj).sum()
            S += w_k * T_k
        return csr_matrix(S)

    def _prop_gprgnn(self) -> csr_matrix:
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        P = diags(1.0 / row_sums).dot(self.A)
        N = self.A.shape[0]
        S = csr_matrix((N, N))
        P_power = identity(N, format='csr')
        for k in range(self.gpr_k + 1):
            S += self.gpr_theta[k] * P_power
            P_power = P.dot(P_power)
        return csr_matrix(S)

    def _prop_jacobiconv(self) -> csr_matrix:
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D_inv = diags(1.0 / row_sums)
        M = self.jacobi_alpha * D_inv.dot(self.A)
        N = self.A.shape[0]
        S = identity(N, format='csr')
        M_power = identity(N, format='csr')
        for _ in range(self.jacobi_iters):
            M_power = M.dot(M_power)
            S += M_power
        return csr_matrix(S)

    def _prop_s2gc(self) -> csr_matrix:
        """Simple Spectral Graph Convolution (SSGConv / S2GC) propagation."""
        N = self.A.shape[0]
        # 1) Add self-loops
        A_hat = self.A + identity(N, format='csr')
        # 2) Symmetric normalization: D^{-1/2} A_hat D^{-1/2}
        row_sums = np.array(A_hat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        norm_A = diags(inv_sqrt).dot(A_hat).dot(diags(inv_sqrt))
        # 3) Build S = alpha * I + (1 - alpha)/K sum_{k=1..K} norm_A^k
        alpha = self.s2gc_alpha
        K = self.s2gc_k
        S = alpha * identity(N, format='csr')
        M_power = identity(N, format='csr')
        for _ in range(K):
            M_power = norm_A.dot(M_power)
            S += (1.0 - alpha) / K * M_power
        return csr_matrix(S)

    def compute_variance(self, method: str = None) -> float:
        m = (method or self.method).lower()
        if m == 'all':
            raise ValueError("Use compute_all for all methods")
        if m not in self.SUPPORTED:
            raise ValueError(f"Unknown method '{m}'")
        func = getattr(self, f"_prop_{m}")
        S = func()
        return float(np.var(S.toarray().flatten()))

    def compute_all(self) -> dict:
        return {m: self.compute_variance(m)
                for m in self.SUPPORTED if m != 'all'}
