import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits
import torch.nn.init as init
from sklearn.neighbors import kneighbors_graph
import numpy as np


class AttentiveLayer(nn.Module):

    def __init__(self, d):
        super(AttentiveLayer, self).__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x @ torch.diag(self.w)


class AttentiveEncoder(nn.Module):

    def __init__(self, nlayers, d, activation='F.relu'):
        super(AttentiveEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(AttentiveLayer(d))
        self.activation = eval(activation)

    def forward(self, h, adj):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = self.activation(h)
        return h


class MLPEncoder(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, use_bn=True, activation='F.relu'):
        super(MLPEncoder, self).__init__()
        self.in_channels = in_channels
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activation = eval(activation)
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def param_init(self):
        # used by sublime
        for layer in self.lins:
            layer.weight = nn.Parameter(torch.eye(self.in_channels))

    def forward(self, feats, adj):
        x = feats
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x