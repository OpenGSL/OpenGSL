import numpy as np
import torch
from opengsl.method.transform import normalize
from torch.optim import Optimizer
from torch.optim.optimizer import required
import scipy.sparse as sp


def connectivity_regularizer(adj):
    if adj.is_sparse:
        adj = adj.to_dense()
    return -1 * torch.log(adj.sum(1) + 1e-12).sum()


def smoothness_regularizer(x, adj, style=None, symmetric=False):
    n = x.shape[0]
    device = x.device
    if adj.is_sparse:
        adj = adj.to_dense()
    if symmetric:
        adj = (adj.t() + adj) / 2
    if style is None:
        L = torch.diag(adj.sum(1)) - adj
    elif style == 'row':
        L = torch.eye(n).to(device) - normalize(adj, 'row')
    elif style == 'symmetric':
        L = torch.eye(n).to(device) - normalize(adj, 'symmetric', add_loop=False)
    else:
        raise KeyError("The normalize style is not provided.")
    return torch.trace(x.T @ L @ x)


def smoothness_regularizer_direct(x, adj):
    assert adj.is_sparse, 'this regularizer if only for big sparse adj.'
    edge = adj.indices()
    value = adj.values()
    h = torch.index_select(x, 0, edge[0]) - torch.index_select(x, 0, edge[1])
    h = (h**2).sum(1)
    return (h * value).sum()


def norm_regularizer(adj, p='fro'):
    if adj.is_sparse:
        adj = adj.to_dense()
    return torch.norm(adj, p)


class PGD(Optimizer):
    """Proximal gradient descent.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining parameter groups
    proxs : iterable
        iterable of proximal operators
    alpha : iterable
        iterable of coefficients for proximal gradient descent
    lr : float
        learning rate
    momentum : float
        momentum factor (default: 0)
    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    dampening : float
        dampening for momentum (default: 0)

    """

    def __init__(self, params, proxs, alphas, lr=required, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)


        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    # param.data.add_(lr, -param.grad.data)
                    # param.data.add_(delta)
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators:
    """Proximal Operators.
    """

    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        """Proximal operator for l1 norm.
        """
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data)-alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        """Proximal operator for nuclear norm (trace norm).
        """
        U, S, V = np.linalg.svd(data.cpu())
        U, S, V = torch.FloatTensor(U).cuda(), torch.FloatTensor(S).cuda(), torch.FloatTensor(V).cuda()
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_truncated(self, data, alpha, k=50):
        indices = torch.nonzero(data).t()
        values = data[indices[0], indices[1]] # modify this based on dimensionality
        data_sparse = sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))
        U, S, V = sp.linalg.svds(data_sparse, k=k)
        U, S, V = torch.FloatTensor(U).cuda(), torch.FloatTensor(S).cuda(), torch.FloatTensor(V).cuda()
        self.nuclear_norm = S.sum()
        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_cuda(self, data, alpha):

        U, S, V = torch.svd(data)
        # self.nuclear_norm = S.sum()
        # print(f"rank = {len(S.nonzero())}")
        self.nuclear_norm = S.sum()
        S = torch.clamp(S-alpha, min=0)
        indices = torch.tensor([range(0, U.shape[0]),range(0, U.shape[0])]).cuda()
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))
        # diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        # print(f"rank_after = {len(diag_S.nonzero())}")
        V = torch.spmm(diag_S, V.t_())
        V = torch.matmul(U, V)
        return V


if __name__ == '__main__':
    from torch_geometric import seed_everything
    seed_everything(135)
    a = torch.rand(100,100)
    print(connectivity_regularizer(a)/100)
    ones_vec = torch.ones(a.size(-1))
    loss = -1 * torch.mm(ones_vec.unsqueeze(0), torch.log(
        torch.mm(a, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / a.shape[-1]
    print(loss)
