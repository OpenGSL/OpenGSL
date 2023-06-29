import torch
import numpy as np
import scipy.sparse as sp
from opengsl.utils.utils import scipy_sparse_to_sparse_tensor ,sparse_tensor_to_scipy_sparse


def normalize(mx, style='symmetric', add_loop=True, sparse=False):
    '''
    Normalize the feature matrix or adj matrix.

    Parameters
    ----------
    mx : torch.tensor
        Feature matrix or adj matrix to normalize.
    style: str
        If set as ``row``, `mx` will be row-wise normalized.
        If set as ``symmetric``, `mx` will be normalized as in GCN.

    add_loop : bool
        Whether to add self loop.
    sparse : bool
        Whether the matrix is stored in sparse form. The returned tensor will be the same form.

    Returns
    -------
    normalized_mx : torch.tensor
        The normalized matrix.

    '''
    if style == 'row':
        return row_nomalize(mx)
    elif style == 'symmetric':
        if sparse:
            return normalize_sp_tensor(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)


def row_nomalize(mx):
    """Row-normalize sparse matrix.
    """
    device = mx.device
    mx = mx.cpu().numpy()
    r_sum = np.array(mx.sum(1))
    r_inv = np.power(r_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.tensor(mx).to(device)
    return mx


def normalize_tensor(adj, add_loop=True):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0]).to(device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A


def normalize_sp_tensor(adj, add_loop=True):
    device = adj.device
    adj = sparse_tensor_to_scipy_sparse(adj)
    adj = normalize_sp_matrix(adj, add_loop)
    adj = scipy_sparse_to_sparse_tensor(adj).to(device)
    return adj


def normalize_sp_matrix(adj, add_loop=True):
    mx = adj + sp.eye(adj.shape[0]) if add_loop else adj
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    new = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return new