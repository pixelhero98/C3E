import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, identity
from torch_geometric.data import Data
from torch_geometric.transforms import GDC
from torch_geometric.utils import gcn_norm, to_scipy_sparse_matrix


def scaled_laplacian(A_csr):
    """Compute scaled Laplacian \tilde{L} = 2L/\lambda_max - I using lambda_max≈2"""
    row_sums = np.array(A_csr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    inv_sqrt = 1.0 / np.sqrt(row_sums)
    D_inv_sqrt = diags(inv_sqrt)
    L = identity(A_csr.shape[0], format='csr') - D_inv_sqrt.dot(A_csr).dot(D_inv_sqrt)
    return 2 * L - identity(A_csr.shape[0], format='csr')


class PropagationVarianceAnalyzer:
    """
    Analyzer for computing variance of propagation matrix for a single chosen method
    or all methods. Supported methods: 'gcn', 'appnp', 'gdc', 'sgc',
    'chebnetii', 'gprgnn', 'jacobiconv', 'all'.

    Accepts input as a PyG Data object (with edge_index, optional edge_weight)
    or as a scipy sparse CSR matrix.

    Default parameters align with original papers.
    """
    SUPPORTED = {'gcn', 'appnp', 'gdc', 'sgc', 'chebnetii', 'gprgnn', 'jacobiconv', 'all'}

    def __init__(
        self,
        data,
        method: str = 'all',
        alpha: float = 0.05,
        sgc_k: int = 2,
        appnp_k: int = 10,
        cheb_k: int = 5,
        cheb_theta: np.ndarray = None,
        gpr_k: int = 10,
        gpr_theta: np.ndarray = None,
        jacobi_iters: int = 10,
        jacobi_alpha: float = 0.5,
    ):
        m = method.lower()
        if m not in self.SUPPORTED:
            raise ValueError(f"Unsupported method '{method}'. Choose from {self.SUPPORTED}.")
        self.method = m
        # Build adjacency CSR.
        if isinstance(data, Data):
            edge_weight = data.edge_weight if 'edge_weight' in data else torch.ones(data.edge_index.size(1))
            self.A = to_scipy_sparse_matrix(
                data.edge_index, edge_weight, num_nodes=data.num_nodes
            ).tocsr()
        elif isinstance(data, csr_matrix):
            self.A = data
        else:
            raise TypeError("data must be a PyG Data or scipy.sparse.csr_matrix")
        # Retain Data for transforms
        self.data = data if isinstance(data, Data) else None
        # Store parameters
        self.alpha = alpha
        self.sgc_k = sgc_k
        self.appnp_k = appnp_k
        self.cheb_k = cheb_k
        self.cheb_theta = cheb_theta if cheb_theta is not None else np.ones(self.cheb_k + 1) / (self.cheb_k + 1)
        self.gpr_k = gpr_k
        self.gpr_theta = gpr_theta if gpr_theta is not None else np.ones(self.gpr_k + 1) / (self.gpr_k + 1)
        self.jacobi_iters = jacobi_iters
        self.jacobi_alpha = jacobi_alpha

    def _prop_gcn(self) -> csr_matrix:
        if self.data is None:
            raise ValueError("GCN requires PyG Data input with edge_index")
        edge_index, edge_weight = gcn_norm(
            self.data.edge_index,
            self.data.edge_weight,
            num_nodes=self.A.shape[0],
            add_self_loops=True,
            normalization='sym'
        )
        return to_scipy_sparse_matrix(edge_index, edge_weight, self.A.shape)

    def _prop_appnp(self) -> csr_matrix:
        row_sums = np.array(self.A.sum(axis=1)).flatten(); row_sums[row_sums == 0] = 1
        P = diags(1.0 / row_sums).dot(self.A)
        M = P.multiply(1 - self.alpha)
        N = self.A.shape[0]
        S = identity(N, format='csr') * self.alpha
        M_power = identity(N, format='csr')
        for _ in range(1, self.appnp_k):
            M_power = M.dot(M_power)
            S += self.alpha * M_power
        S += M.dot(M_power)
        return csr_matrix(S)

    def _prop_gdc(self) -> csr_matrix:
        if self.data is None:
            raise ValueError("GDC requires PyG Data input with edge_index")
        transform = GDC(self_loop=True, normalization_in='sym', alpha=self.alpha)
        data_gdc = transform(self.data)
        return to_scipy_sparse_matrix(data_gdc.edge_index, data_gdc.edge_weight, self.A.shape)

    def _prop_sgc(self) -> csr_matrix:
        N = self.A.shape[0]
        A_hat = self.A + identity(N, format='csr')
        row_sums = np.array(A_hat.sum(axis=1)).flatten(); row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        norm_A = diags(inv_sqrt).dot(A_hat).dot(diags(inv_sqrt))
        M = norm_A
        for _ in range(1, self.sgc_k): M = M.dot(norm_A)
        return csr_matrix(M)

    def _prop_chebnetii(self) -> csr_matrix:
        L_tilde = scaled_laplacian(self.A)
        N = self.A.shape[0]
        j = np.arange(self.cheb_k + 1)
        xj = np.cos((2 * j + 1) * np.pi / (2 * (self.cheb_k + 1)))
        T_prev = identity(N, format='csr'); T_curr = L_tilde
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
        row_sums = np.array(self.A.sum(axis=1)).flatten(); row_sums[row_sums == 0] = 1
        P = diags(1.0 / row_sums).dot(self.A)
        N = self.A.shape[0]
        S = csr_matrix((N, N)); P_power = identity(N, format='csr')
        for k in range(self.gpr_k + 1):
            S += self.gpr_theta[k] * P_power
            P_power = P.dot(P_power)
        return csr_matrix(S)

    def _prop_jacobiconv(self) -> csr_matrix:
        row_sums = np.array(self.A.sum(axis=1)).flatten(); row_sums[row_sums == 0] = 1
        D_inv = diags(1.0 / row_sums)
        M = self.jacobi_alpha * D_inv.dot(self.A)
        N = self.A.shape[0]
        S = identity(N, format='csr'); M_power = identity(N, format='csr')
        for _ in range(self.jacobi_iters):
            M_power = M.dot(M_power)
            S += M_power
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
        results = {}
        for m in self.SUPPORTED - {'all'}:
            results[m] = self.compute_variance(m)
        return results
