import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, identity
from torch_geometric.data import Data
from torch_geometric.transforms import GDC
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix


def scaled_laplacian(A_csr: csr_matrix) -> csr_matrix:
    """Compute scaled Laplacian Ṽ = 2L/λₘₐₓ – I (using λₘₐₓ ≈ 2)."""
    row_sums = np.array(A_csr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    inv_sqrt = 1.0 / np.sqrt(row_sums)
    D_inv_sqrt = diags(inv_sqrt)
    L = identity(A_csr.shape[0], format='csr') - D_inv_sqrt @ A_csr @ D_inv_sqrt
    return 2 * L - identity(A_csr.shape[0], format='csr')


class PropagationVarianceAnalyzer:
    """
    Computes variance of the propagation matrix for various GNN-style
    propagation schemes. Supports: 'gcn', 'appnp', 'gdc', 'sgc',
    'chebnetii', 'gprgnn', 'jacobiconv', 's2gc', or 'all'.

    GDC (diffusion) options:
      - diffusion via 'ppr' or 'heat'
      - sparsification via 'topk' or 'threshold'
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
        # general diffusion teleport prob
        alpha: float = 0.05,
        # APPNP params
        appnp_k: int = 10,
        appnp_alpha: float = 0.1,
        # SGC
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
        # S2GC
        s2gc_k: int = 1,
        s2gc_alpha: float = 0.5,
        # GDC diffusion
        gdc_method: str = 'ppr',           # 'ppr' or 'heat'
        heat_t: float = 1.0,
        gdc_diffusion_eps: float = 1e-4,   # for approximate PPR
        # GDC sparsification
        gdc_spars_method: str = 'topk',    # 'topk' or 'threshold'
        gdc_avg_degree: int = 64,
        gdc_threshold_eps: float = None,
        gdc_exact: bool = False            # override to force exact
    ):
        self.method = method.lower()
        if self.method not in self.SUPPORTED:
            raise ValueError(f"Unsupported method '{method}'. Choose from {self.SUPPORTED}.")

        # Build adjacency CSR
        if isinstance(data, Data):
            edge_weight = getattr(data, 'edge_weight',
                                  torch.ones(data.edge_index.size(1)))
            self.A = to_scipy_sparse_matrix(
                data.edge_index, edge_weight, num_nodes=data.num_nodes
            ).tocsr()
        elif isinstance(data, csr_matrix):
            self.A = data
            data = None
        else:
            raise TypeError("`data` must be a PyG Data or scipy.sparse.csr_matrix")
        self.data = data

        # Store common params
        self.alpha        = alpha
        self.appnp_k      = appnp_k
        self.appnp_alpha  = appnp_alpha
        self.sgc_k        = sgc_k
        self.cheb_k       = cheb_k
        self.cheb_theta   = (
            cheb_theta
            if cheb_theta is not None
            else np.ones(self.cheb_k + 1) / (self.cheb_k + 1)
        )
        self.gpr_k        = gpr_k
        self.gpr_theta    = (
            gpr_theta
            if gpr_theta is not None
            else np.ones(self.gpr_k + 1) / (self.gpr_k + 1)
        )
        self.jacobi_iters = jacobi_iters
        self.jacobi_alpha = jacobi_alpha
        self.s2gc_k       = s2gc_k
        self.s2gc_alpha   = s2gc_alpha

        # GDC diffusion params
        self.gdc_method           = gdc_method.lower()
        if self.gdc_method not in {'ppr', 'heat'}:
            raise ValueError("gdc_method must be 'ppr' or 'heat'")
        self.heat_t               = heat_t
        self.gdc_diffusion_eps    = gdc_diffusion_eps

        # GDC sparsification params
        self.gdc_spars_method     = gdc_spars_method.lower()
        if self.gdc_spars_method not in {'topk', 'threshold'}:
            raise ValueError("gdc_spars_method must be 'topk' or 'threshold'")
        self.gdc_avg_degree       = gdc_avg_degree
        self.gdc_threshold_eps    = gdc_threshold_eps
        self.gdc_exact            = gdc_exact

    def _prop_gcn(self) -> csr_matrix:
        if self.data is None:
            raise ValueError("GCN requires PyG Data with edge_index")
        edge_index, edge_weight = gcn_norm(
            self.data.edge_index,
            getattr(self.data, 'edge_weight', None),
            num_nodes=self.A.shape[0],
            improved=False,
            add_self_loops=True
        )
        return to_scipy_sparse_matrix(edge_index, edge_weight,
                                      num_nodes=self.A.shape[0])

    def _prop_appnp(self) -> csr_matrix:
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
        return csr_matrix(S)

    def _prop_gdc(self) -> csr_matrix:
        if self.data is None:
            raise ValueError("GDC requires PyG Data with edge_index")

        N = self.A.shape[0]

        # diffusion kwargs (always include eps for PPR)
        if self.gdc_method == 'ppr':
            diffusion_kwargs = {
                'method': 'ppr',
                'alpha': self.alpha,
                'eps': self.gdc_diffusion_eps,
            }
        else:  # heat
            diffusion_kwargs = {
                'method': 'heat',
                't': self.heat_t,
            }

        # sparsification kwargs
        if self.gdc_spars_method == 'topk':
            spars_kwargs = {
                'method': 'topk',
                'k': self.gdc_avg_degree,
                'dim': 1,
            }
        else:  # threshold
            spars_kwargs = {
                **({'eps': self.gdc_threshold_eps}
                   if self.gdc_threshold_eps is not None
                   else {'avg_degree': self.gdc_avg_degree})
            }
            spars_kwargs['method'] = 'threshold'

        # force exact for topk or heat if needed
        effective_exact = (
            self.gdc_exact
            or self.gdc_spars_method == 'topk'
            or diffusion_kwargs['method'] == 'heat'
        )

        transform = GDC(
            self_loop_weight=1.0,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=diffusion_kwargs,
            sparsification_kwargs=spars_kwargs,
            exact=effective_exact,
        )
        data_gdc = transform(self.data)

        return to_scipy_sparse_matrix(
            data_gdc.edge_index,
            data_gdc.edge_attr,
            num_nodes=N
        ).tocsr()

    def _prop_sgc(self) -> csr_matrix:
        N = self.A.shape[0]
        A_hat = self.A + identity(N, format='csr')
        row_sums = np.array(A_hat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        norm_A = diags(inv_sqrt) @ A_hat @ diags(inv_sqrt)
        M = norm_A
        for _ in range(1, self.sgc_k):
            M = M @ norm_A
        return csr_matrix(M)

    def _prop_chebnetii(self) -> csr_matrix:
        L_tilde = scaled_laplacian(self.A)
        N = self.A.shape[0]
        j = np.arange(self.cheb_k + 1)
        xj = np.cos((2*j + 1) * np.pi / (2*(self.cheb_k + 1)))
        T_prev = identity(N, format='csr')
        T_curr = L_tilde
        S = csr_matrix((N, N))
        for k in range(self.cheb_k + 1):
            if k == 0:
                T_k = T_prev
            elif k == 1:
                T_k = T_curr
            else:
                T_k = 2 * L_tilde @ T_curr - T_prev
                T_prev, T_curr = T_curr, T_k
            Tk_xj = np.cos(k * np.arccos(xj))
            w_k = 2.0 / (self.cheb_k + 1) * (self.cheb_theta * Tk_xj).sum()
            S += w_k * T_k
        return csr_matrix(S)

    def _prop_gprgnn(self) -> csr_matrix:
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
        return csr_matrix(S)

    def _prop_s2gc(self) -> csr_matrix:
        N = self.A.shape[0]
        A_hat = self.A + identity(N, format='csr')
        row_sums = np.array(A_hat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        norm_A = diags(inv_sqrt) @ A_hat @ diags(inv_sqrt)
        alpha = self.s2gc_alpha
        K = self.s2gc_k
        S = alpha * identity(N, format='csr')
        M_power = identity(N, format='csr')
        for _ in range(K):
            M_power = norm_A @ M_power
            S += (1.0 - alpha) / K * M_power
        return csr_matrix(S)

    def compute_variance(self, method: str = None) -> float:
        m = (method or self.method).lower()
        if m == 'all':
            raise ValueError("Use compute_all() to get all methods")
        if m not in self.SUPPORTED:
            raise ValueError(f"Unknown method '{m}'")
        func = getattr(self, f"_prop_{m}")
        S = func()
        return float(np.var(S.toarray().ravel()))

    def compute_all(self) -> dict:
        return {
            m: self.compute_variance(m)
            for m in self.SUPPORTED if m != 'all'
        }
