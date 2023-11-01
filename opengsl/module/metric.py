import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits
import torch.nn.init as init
from sklearn.neighbors import kneighbors_graph
import numpy as np


class WeightedCosine(nn.Module):
    '''
    Weighted cosine to generate pairwise similarities from given node embeddings.

    Parameters
    ----------
    d_in : int
        Dimensions of input features.
    num_pers : int
        Number of multi heads.
    weighted : bool
        Whether to use weighted cosine. cosine will be used if set to `None`.
    normalize : bool
        Whetehr to use normalize before multiplication.
    '''

    def __init__(self, d_in, num_pers=16, weighted=True, normalize=True):
        super(WeightedCosine, self).__init__()
        self.normalize = normalize
        self.w = None
        if weighted:
            self.w = nn.Parameter(torch.FloatTensor(num_pers, d_in))
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.w)

    def forward(self, x, y=None, non_negative=False):
        '''
        Given two groups of node embeddings, calculate the pairwise similarities.

        Parameters
        ----------
        x : torch.tensor
            Input features.
        y : torch.tensor
            Input features. ``x`` will be used if set to `None`.
        non_negative : bool
            Whether to mask negative elements.

        Returns
        -------
        adj : torch.tensor
            Pairwise similarities.
        '''
        if y is None:
            y = x
        context_x = x.unsqueeze(0)
        context_y = y.unsqueeze(0)
        if self.w is not None:
            expand_weight_tensor = self.w.unsqueeze(1)
            context_x = context_x * expand_weight_tensor
            context_y = context_y * expand_weight_tensor
        if self.normalize:
            context_x = F.normalize(context_x, p=2, dim=-1)
            context_y = F.normalize(context_y, p=2, dim=-1)
        adj = torch.matmul(context_x, context_y.transpose(-1, -2)).mean(0)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj


class Cosine(nn.Module):
    '''
    Cosine to generate pairwise similarities from given node embeddings.
    '''
    def __init__(self):
        super(Cosine, self).__init__()
        pass

    def forward(self, x, y=None, non_negative=False):
        '''
        Given two groups of node embeddings, calculate the pairwise similarities.

        Parameters
        ----------
        x : torch.tensor
            Input features.
        y : torch.tensor
            Input features. ``x`` will be used if set to `None`.
        non_negative : bool
            Whether to mask negative elements.

        Returns
        -------
        adj : torch.tensor
            Pairwise similarities.
        '''
        if y is None:
            y = x
        context_x = F.normalize(x, p=2, dim=-1)
        context_y = F.normalize(y, p=2, dim=-1)
        adj = torch.matmul(context_x, context_y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj


class InnerProduct(nn.Module):
    '''
    InnerProduct to generate pairwise similarities from given node embeddings.
    '''

    def __init__(self):
        super(InnerProduct, self).__init__()
        pass

    def forward(self, x, y=None, non_negative=False):
        '''
        Given two groups of node embeddings, calculate the pairwise similarities.

        Parameters
        ----------
        x : torch.tensor
            Input features.
        y : torch.tensor
            Input features. `x` will be used if set to ``None``.
        non_negative : bool
            Whether to mask negative elements.

        Returns
        -------
        adj : torch.tensor
            Pairwise similarities.
        '''
        if y is None:
            y = x
        adj = torch.matmul(x, y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj


# class GeneralizedMetric(nn.Module):
#
#     def __init__(self, d_in, num_pers=16, normalize=True):
#         super(GeneralizedMetric, self).__init__()
#         self.normalize = normalize
#         # self.Q = nn.Parameter(torch.FloatTensor(num_pers, d_in, d_in))
#         self.Q = nn.Parameter(torch.eye(d_in).unsqueeze(0).repeat(num_pers,1,1))
#         # self.Q = torch.eye(d_in).unsqueeze(0).repeat(num_pers,1,1).to('cuda:0')
#         # self.reset_parameters()
#
#     def reset_parameters(self):
#         init.xavier_uniform_(self.Q)
#
#     def forward(self, x, y=None, non_negative=False):
#         Q = F.softmax(self.Q, dim=-1)
#         n_h = self.Q.shape[0]
#         if y is None:
#             y = x
#         context_x = x.unsqueeze(0)
#         context_y = y.unsqueeze(0)
#         if self.normalize:
#             context_x = F.normalize(context_x, p=2, dim=-1)
#             context_y = F.normalize(context_y, p=2, dim=-1)
#         adj = torch.bmm(torch.bmm(context_x.repeat(n_h, 1, 1), Q), context_y.transpose(-1, -2).repeat(n_h, 1, 1)).mean(0)
#         if non_negative:
#             mask = (adj > 0).detach().float()
#             adj = adj * mask + 0 * (1 - mask)
#         return adj


class FGP(nn.Module):
    def __init__(self, n, nonlinear=None, init_adj=None):
        super(FGP, self).__init__()
        self.Adj = nn.Parameter(torch.FloatTensor(n, n))
        self.nonlinear = lambda adj: F.elu(adj) + 1
        if nonlinear:
            self.nonlinear = eval(nonlinear)
        if init_adj:
            self.init_estimation(init_adj)

    def reset_parameters(self, features, k, metric, i):
        adj = kneighbors_graph(features, k, metric=metric)
        adj = np.array(adj.todense(), dtype=np.float32)
        adj += np.eye(adj.shape[0])
        adj = adj * i - i
        self.Adj.data.copy_(torch.tensor(adj))

    def init_estimation(self, adj):
        self.Adj.data.copy_(adj)

    def forward(self, x):
        return self.nonlinear(self.Adj)


class GeneralizedMahalanobis(nn.Module):
    '''
    Metric from `"Adaptive Graph Convolutional Neural Networks" <http://arxiv.org/abs/1801.03226>`_ paper
    '''
    def __init__(self, d_in, sigma=1):
        super(GeneralizedMahalanobis, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(d_in, d_in))
        self.sigma = sigma

    def forward(self, x, y=None, edge=None):
        device = x.device
        if y is None:
            y = x
        M = self.W @ self.W.T
        if edge:
            d = torch.index_select(x, 0, edge[0]) - torch.index_select(y, 0, edge[1])
            D = torch.sqrt(((d @ M) * d).sum(1))
            D = torch.exp(-D / (2*self.sigma**2))
            return torch.sparse.FloatTensor(edge, D, [x.shape[0], y.shape[0]]).to(device)
        else:
            D = torch.zeros(x.shape[0], y.shape[0])
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    d = x[i] - y[j]
                    D[i, j] = d @ M @d.T
            D = torch.exp(-D / (2 * self.sigma ** 2))
            return D



if __name__ == '__main__':
    from torch_geometric import seed_everything
    seed_everything(42)
    f = WeightedCosine(3, 2, True)
    x = torch.rand(10,3)
    print(f(x).shape)