import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothness_regularizer(x, adj, sparse=False, normalize=None):
    n = x.shape[0]
    if sparse:
        adj = adj.to_dense()
    L = torch.diag(adj.sum(1)) - adj
    return torch.trace(x.T @ L @ x)


def frobenius_regularizer(adj):
    return torch.norm(adj, p='fro')
