import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from opengsl.utils.utils import scipy_sparse_to_sparse_tensor, sparse_tensor_to_scipy_sparse


def normalize(mx, style='symmetric', add_loop=True, p=None):
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

    Returns
    -------
    normalized_mx : torch.tensor
        The normalized matrix.

    '''
    if style == 'row':
        if mx.is_sparse:
            return row_normalize_sp(mx)
        else:
            return row_nomalize(mx)
    elif style == 'symmetric':
        if mx.is_sparse:
            return normalize_sp_tensor_tractable(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)
    elif style == 'softmax':
        if mx.is_sparse:
            return torch.sparse.softmax(mx, dim=-1)
        else:
            return F.softmax(mx, dim=-1)
    elif style == 'row-norm':
        assert p is not None
        if mx.is_sparse:
            # TODO
            pass
        else:
            return F.normalize(mx, dim=-1, p=p)
    else:
        raise KeyError("The normalize style is not provided.")


def row_nomalize(mx):
    """Row-normalize sparse matrix.
    """
    # device = mx.device
    # mx = mx.cpu().numpy()
    # r_sum = np.array(mx.sum(1))
    # r_inv = np.power(r_sum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)
    # mx = torch.tensor(mx).to(device)

    r_sum = mx.sum(1)
    r_inv = r_sum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx

    return mx


def row_normalize_sp(mx):
    inv_sqrt_degree = 1. / (torch.sparse.sum(mx, dim=1).values() + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def normalize_sp_tensor_tractable(adj, add_loop=True):
    n = adj.shape[0]
    device = adj.device
    if add_loop:
        adj = adj + torch.eye(n, device=device).to_sparse()
        adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


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


def symmetry(adj):
    if adj.is_sparse:
        n = adj.shape[0]
        adj_t = torch.sparse.FloatTensor(adj.indices()[[1,0]], adj.values(), [n, n])
        return (adj_t + adj).coalesce() / 2
    else:
        return (adj.t() + adj) / 2


def knn(adj, K):
    device = adj.device
    values, indices = adj.topk(k=int(K), dim=-1)
    assert torch.max(indices) < adj.shape[1]
    mask = torch.zeros(adj.shape).to(device)
    mask[torch.arange(adj.shape[0]).view(-1, 1), indices] = 1.
    mask.requires_grad = False
    new_adj = adj * mask
    return new_adj


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def apply_non_linearity(adj, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(adj * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(adj)
    elif non_linearity == 'none':
        return adj
    else:
        raise KeyError('We dont support the non-linearity yet')


if __name__ == '__main__':
    from torch_geometric import seed_everything
    seed_everything(42)
    # adj = torch.rand(5, 5).to_sparse()
    # adj = torch.sparse.FloatTensor(torch.tensor([[0,0,1,1,2,2,3,3,4],[1,2,3,4,0,1,2,3,3]]), torch.tensor([1,1,1,1,1,1,1,1,1]), [5,5])
    adj = torch.rand(3,3)
    print(adj)
    print(knn(adj, 2))