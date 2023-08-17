import torch
import torch.nn as nn
import torch.nn.functional as F
import math


'''
This GCN is only used for IGDL for its changeable dropout.
'''


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, with_bias=True, batch_norm=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def init_params(self):
        # initialize weights with xavier uniform and biases with all zeros.
        # This is more recommended than the upper one.
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, input, adj, batch_norm=True):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)
        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, n_layers=2, dropout=0.5, with_bias=True, batch_norm=False):

        super(GCN, self).__init__()

        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias, batch_norm=batch_norm))
        for i in range(n_layers-2):
            self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias, batch_norm=batch_norm))
        self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias, batch_norm=False))
        self.dropout = dropout
        self.with_bias = with_bias
        self.batch_norm = batch_norm

    def forward(self, x, adj, dropout=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x, adj))
            x = F.dropout(x, dropout if dropout else self.dropout, training=self.training)
        output = self.layers[-1](x, adj)
        return x, output.squeeze(1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for i in range(self.n_layers):
            self.layers[i].reset_parameters()