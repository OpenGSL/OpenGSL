import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


'''
This GCN is used to methods that need to modify the adj of graph because modifying the adj in dgl graph is less convenient.
'''


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input, adj, batch_norm=True):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        # output = torch.sparse.mm(adj, support)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, n_layers=2, dropout=0.5, input_dropout=0.0, use_linaer=False, norm=False):

        super(GCN, self).__init__()

        self.use_linear = use_linaer
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        if norm:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            in_hidden = nhid if i > 0 else nfeat
            out_hidden = nhid if i < n_layers - 1 else nclass
            bias = not norm or i == n_layers - 1

            self.convs.append(GraphConvolution(in_hidden, out_hidden, with_bias=bias))
            if use_linaer:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                if norm:
                    self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.input_drop(x)
        for i, layer in enumerate(self.convs):
            conv = layer(x,adj)
            if self.use_linear:
                linear = self.linear[i](x)
                x = conv+linear
            else:
                x = conv
            if i < self.n_layers - 1:
                if self.norms is not None:
                    x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
                mid = x
        return mid, x.squeeze(1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for i in range(self.n_layers):
            self.layers[i].reset_parameters()