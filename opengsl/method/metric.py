import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits


class OneLayerNN(torch.nn.Module):

    def __init__(self, d_in, d_out, p_dropout=0):
        super(OneLayerNN, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(d_in, d_out))
        self.a = nn.Parameter(torch.FloatTensor(d_out, 1))
        self.dropout = nn.Dropout(p_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.W)
        inits.glorot(self.a)

    def forward(self, x, edge, return_h=True):
        device = x.device
        n = x.shape[0]
        x = self.dropout(x)
        h = x @ self.W
        d = torch.abs(torch.index_select(h, 0, edge[0]) - torch.index_select(h, 0, edge[1]))
        values = F.relu(d @ self.a).squeeze(1)
        adj = torch.sparse.FloatTensor(edge, values, [n,n]).to(device)
        adj = torch.sparse.softmax(adj, 1)
        if return_h:
            return adj.coalesce(), h
        else:
            return adj.coalesce()