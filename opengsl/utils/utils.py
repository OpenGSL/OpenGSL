import dgl.random
import torch
import os
import numpy as np
import scipy.sparse as sp
import random


def accuracy(labels, logits):
    '''
    Compute the accuracy score given true labels and predicted labels.

    Parameters
    ----------
    labels: np.array
        Ground truth labels.
    logits : np.array
        Predicted labels.

    Returns
    -------
    accuracy : np.float
        The Accuracy score.

    '''
    return np.sum(logits.argmax(1)==labels)/len(labels)


def scipy_sparse_to_sparse_tensor(sparse_mx):
    '''
    Convert a scipy sparse matrix to a torch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse_matrix
        Sparse matrix to convert.

    Returns
    -------
    sparse_tensor: torch.Tensor in sparse form
        A tensor stored in sparse form.
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_tensor_to_scipy_sparse(sparse_tensor):
    '''
    Convert a torch sparse tensor to a scipy sparse matrix.

    Parameters
    ----------
    sparse_tensor : torch.Tensor in sparse form
        A tensor stored in sparse form to convert.

    Returns
    -------
    sparse_mx : scipy.sparse_matrix
        Sparse matrix.

    '''
    sparse_tensor = sparse_tensor.cpu()
    row = sparse_tensor.coalesce().indices()[0].numpy()
    col = sparse_tensor.coalesce().indices()[1].numpy()
    values = sparse_tensor.coalesce().values().numpy()
    return sp.coo_matrix((values, (row, col)), shape=sparse_tensor.shape)


def set_seed(seed):
    '''
    Set seed to make sure the results can be repetitive.

    Parameters
    ----------
    seed : int
        Random seed to set.
    '''
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_node_homophily(label, adj):
    '''
    Calculate the node homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    homophily : torch.float
        The node homophily of the graph.

    '''
    label = label.cpu().numpy()
    adj = adj.cpu().numpy()
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = (np.multiply((label == label.T), adj)).sum(axis=1)
    d = adj.sum(axis=1)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1. / d[i])
    return np.mean(homos)


def get_edge_homophily(label, adj):
    '''
    Calculate the node homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    homophily : torch.float
        The edge homophily of the graph.

    '''
    num_edge = adj.sum()
    cnt = 0
    for i, j in adj.nonzero():
        if label[i] == label[j]:
            cnt += adj[i, j]
    return cnt/num_edge


def get_homophily(label, adj, type='node', fill=None):
    '''
    Calculate node or edge homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.
    type : str
        This decides whether to calculate node homo or edge homo.
    fill : str
        The value to fill in the diagonal of `adj`. If set to `None`, the operation won't be done.

    Returns
    -------
    homophily : np.float
        The node or edge homophily of a graph.

    '''
    if fill:
        np.fill_diagonal(adj, fill)
    return eval('get_'+type+'_homophily(label, adj)')


def get_adjusted_homophily(_label, adj):
    '''
    Calculate adjusted homophily of a graph.

    Parameters
    ----------
    _label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    homophily : np.float
        The adjusted homophily of a graph.

    '''
    label = _label.long()
    labels = label.max() + 1
    d = adj.sum(1)
    E = d.sum()
    D = torch.zeros(labels)
    for i in range(adj.shape[0]):
        D[label[i]] += d[i]

    h_edge = get_edge_homophily(label, adj)
    sum_pk = ((D / E) ** 2).sum()

    return (h_edge - sum_pk) / (1 - sum_pk)


def get_label_informativeness(_label, adj):
    '''
    Calculate label informativeness of a graph.

    Parameters
    ----------
    _label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    label_informativeness : np.float
        The label informativeness of a graph.
    '''
    label = _label.long()
    labels = label.max() + 1
    LI_1 = 0
    LI_2 = 0

    p = torch.zeros((labels, labels))
    for i, j in adj.nonzero():
        p[label[i]][label[j]] = p[label[i]][label[j]] + adj[i][j]

    d = adj.sum(1)
    E = d.sum()
    D = torch.zeros(labels)

    for i in range(adj.shape[0]):
        D[label[i]] = D[label[i]] + d[i]

    for i in range(labels):
        for j in range(labels):
            p[i][j] = p[i][j] / E

    p_ = D / E
    LI_2 = (p_ * torch.log(p_)).sum()
    for i in range(labels):
        for j in range(labels):
            if (p[i][j] != 0):
                LI_1 += p[i][j] * torch.log(p[i][j] / (p_[i] * p_[j]))

    return -LI_1 / LI_2

def one_hot(y):
    device = y.device
    c = y.max() + 1
    e = torch.eye(c)
    return e[y].to(device)