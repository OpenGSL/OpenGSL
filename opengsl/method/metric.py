import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits
import torch.nn.init as init
from sklearn.neighbors import kneighbors_graph
import numpy as np


class WeightedCosine(nn.Module):

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


class Cosine:

    def __init__(self):
        pass

    def __call__(self, x, y=None, non_negative=False):
        if y is None:
            y = x
        context_x = F.normalize(x, p=2, dim=-1)
        context_y = F.normalize(y, p=2, dim=-1)
        adj = torch.matmul(context_x, context_y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj


class InnerProduct:

    def __init__(self):
        pass

    def __call__(self, x, y=None, non_negative=False):
        if y is None:
            y = x
        adj = torch.matmul(x, y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj


if __name__ == '__main__':
    from torch_geometric import seed_everything
    seed_everything(42)
    f = WeightedCosine(10, 0, False, False)
    x = torch.rand(3,10)
    print(f(x))
    print(x @ x.T)