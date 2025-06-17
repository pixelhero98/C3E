import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from torch_geometric.data import Data
from torch_geometric.transforms import GDC
from torch_geometric.utils import gcn_norm, to_scipy_sparse_matrix


class PropagationVarianceAnalyzer:
    """
    Compute propagation matrices (S_l) variances for GCN, APPNP, GDC, and SGC methods.
    """
    def __init__(self, A: coo_matrix, alpha: float = 0.1, sgc_k: int = 2, appnp_k: int = 10):
        # Store adjacency as CSR
        self.A = A.tocsr()
        # Build PyG Data for GCN/GDC
        row, col, data = A.row, A.col, A.data
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_weight = torch.tensor(data, dtype=torch.float)
        self.data = Data(edge_index=edge_index, edge_weight=edge_weight)
        self.alpha = alpha
        self.sgc_k = sgc_k
        self.appnp_k = appnp_k

    def prop_gcn(self) -> csr_matrix:
        """GCN propagation via symmetric normalization and self-loops"""
        edge_index, edge_weight = gcn_norm(
            self.data.edge_index,
            self.data.edge_weight,
            num_nodes=self.A.shape[0],
            add_self_loops=True,
            normalization='sym'
        )
        return to_scipy_sparse_matrix(edge_index, edge_weight, self.A.shape)

    def prop_appnp(self) -> csr_matrix:
        """APPNP propagation via iterative K-step PPR approximation"""
        # Build transition matrix P = D^{-1} A
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D_inv = diags(1.0 / row_sums)
        P = D_inv.dot(self.A)
        # Compute M = (1 - alpha) * P
        M = P.multiply(1 - self.alpha)
        N = self.A.shape[0]
        # Initialize S = (1-alpha)^K P^K + alpha * sum_{i=0 to K-1}( (1-alpha)^i P^i )
        # Compute P^i iteratively
        S_sum = self.alpha * identity(N, format='csr')  # i=0 term: alpha * I
        M_power = identity(N, format='csr')  # M^0
        for i in range(1, self.appnp_k):
            M_power = M.dot(M_power)  # M^i
            S_sum = S_sum + self.alpha * M_power
        # Compute residual term (1-alpha)^K P^K
        M_power = M.dot(M_power)  # M^K
        S = csr_matrix(M_power + S_sum)
        return S

    def prop_gdc(self) -> csr_matrix:
        """GDC propagation via Personalized PageRank diffusion"""
        transform = GDC(self_loop=True, normalization_in='sym', alpha=self.alpha)
        data_gdc = transform(self.data)
        return to_scipy_sparse_matrix(
            data_gdc.edge_index,
            data_gdc.edge_weight,
            self.A.shape
        )

    def prop_sgc(self) -> csr_matrix:
        """SGC propagation: K-th power of symmetric-normalized adjacency"""
        # Symmetric normalization
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        D_inv_sqrt = diags(inv_sqrt)
        norm_A = D_inv_sqrt.dot(self.A).dot(D_inv_sqrt)
        # Raise to k-th power
        M = norm_A
        for _ in range(1, self.sgc_k):
            M = M.dot(norm_A)
        return csr_matrix(M)

    def compute_all(self) -> list:
        """Compute S_l for [GCN, APPNP, GDC, SGC]"""
        return [
            self.prop_gcn(),
            self.prop_appnp(),
            self.prop_gdc(),
            self.prop_sgc()
        ]

    def compute_variances(self) -> np.ndarray:
        """Flatten each S_l and compute its variance"""
        S_list = self.compute_all()
        vars_list = []
        for S in S_list:
            dense = S.toarray()
            vars_list.append(np.var(dense.flatten()))
        return np.array(vars_list)
        print(f"{name}: variance = {v:.6f}")
