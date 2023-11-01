import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
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
    '''
    Attentive encoder used in `"Towards Unsupervised Deep Graph Structure Learning" <https://arxiv.org/abs/2201.06367>`_ paper.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    d : int
        Dimensions of input features.
    activation: str
        Specify the activation function used.
    '''

    def __init__(self, n_layers, d, activation='F.relu'):
        super(AttentiveEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(AttentiveLayer(d))
        self.activation = eval(activation)

    def forward(self, x, adj=None):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.

        Returns
        -------
        x : torch.tensor
            Encoded features.
        '''
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (len(self.layers) - 1):
                x = self.activation(x)
        return x


class MLPEncoder(nn.Module):
    '''
    Adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py

    Parameters
    ----------
    in_channels : int
        Dimensions of input features.
    hidden_channels : int
        Dimensions of hidden representations.
    out_channels : int
        Dimensions of output representations.
    n_layers : int
        Number of layers.
    dropout : float
        Dropout rate.
    use_bn : bool
        Whether to use batch norm.
    activation: str
        Specify the activation function used.
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 dropout=.5, use_bn=True, activation='F.relu'):
        super(MLPEncoder, self).__init__()
        self.in_channels = in_channels
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activation = eval(activation)
        if n_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(n_layers - 2):
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

    def forward(self, x, adj=None):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        Returns
        -------
        x : torch.tensor
            Encoded features.
        '''
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, n_linear=1, bias=True, spmm_type=1, act='F.relu',
                 last_layer=False, weight_initializer=None, bias_initializer=None):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.ModuleList()
        self.mlp.append(Linear(in_features, out_features, bias=bias, weight_initializer=weight_initializer, bias_initializer=bias_initializer))
        for i in range(n_linear-1):
            self.mlp.append(Linear(out_features, out_features, bias=bias, weight_initializer=weight_initializer, bias_initializer=bias_initializer))
        self.dropout = dropout
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]
        self.act = eval(act)
        self.last_layer = last_layer

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        x = self.spmm(adj, input)
        for i in range(len(self.mlp)-1):
            x = self.mlp[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)
        if not self.last_layer:
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionDiagLayer(nn.Module):
    '''
    A GCN convolution layer of diagonal matrix multiplication
    '''
    def __init__(self, input_size):
        super(GraphConvolutionDiagLayer, self).__init__()
        self.W = torch.nn.Parameter(torch.ones(input_size))

    def forward(self, x, adj):
        hidden = x @ torch.diag(self.W)
        output = torch.sparse.mm(adj, hidden)
        return output


class GCNDiagEncoder(nn.Module):
    '''
    Encoder used in `"Graph-Revised Convolutional Network" <https://arxiv.org/abs/1911.07123>`_ paper.

    Parameters
    ----------
    n_layers: int
        Number of layers.
    d : int
        Dimensions of input features.
    activation: str
        Specify the activation function used.
    '''
    def __init__(self, n_layers, d, activation='F.tanh'):
        super(GCNDiagEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GraphConvolutionDiagLayer(d))
        self.activation = eval(activation)

    def forward(self, x, adj):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        Returns
        -------
        x : torch.tensor
            Encoded features.

        '''
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != (len(self.layers) - 1):
                x = self.activation(x)
        return x


class GCNEncoder(nn.Module):
    '''
    GCN encoder.

    Parameters
    ----------
    nfeat : int
        Dimensions of input features.
    nhid : int
        Dimensions of hidden representations.
    nclass : int
        Dimensions of output representations.
    n_layers : int
        Number of layers.
    dropout : float
        Dropout rate.
    input_dropout : float
        Dropout rate on the infut at first. Only used when `input_layer` is `True`.
    norm : str
        Specify batchnorm or layernorm. The operation won't be done if set to `None`.
    n_linear : int
        Number of linear layers in a GCN layer.
    spmm_type: int
        Specify the multiply funtion used between adj and x.
    act : str
        Specify the activation function used.
    input_layer : bool
        Whether to set a seperate linear layer for input.
    output_layer: bool
        Whether to set a seperate linear layer for output.
    weight_initializer: str
        Specify the way to initialize the weights.
    bias_initializer: str
        Specify the way to initialize the bias.
    bias : bool
        Whether to add bias to linear transform in GCN.
    '''
    def __init__(self, nfeat, nhid, nclass, n_layers=2, dropout=0.5, input_dropout=0.0, norm=None, n_linear=1,
                 spmm_type=0, act='F.relu', input_layer=False, output_layer=False, weight_initializer=None,
                 bias_initializer=None, bias=True):

        super(GCNEncoder, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.n_linear = n_linear
        self.norm = norm
        if norm:
            self.norm_type = eval('nn.'+norm['norm_type'])
        self.act = eval(act)
        if input_layer:
            self.input_linear = nn.Linear(in_features=nfeat, out_features=nhid)
            self.input_drop = nn.Dropout(input_dropout)
        if output_layer:
            self.output_linear = nn.Linear(in_features=nhid, out_features=nclass)
            self.output_normalization = self.norm_type(nhid)
        self.convs = nn.ModuleList()
        if self.norm:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = nfeat
            else:
                in_hidden = nhid
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = nclass
            else:
                out_hidden = nhid

            self.convs.append(GraphConvolutionLayer(in_hidden, out_hidden, dropout, n_linear, spmm_type=spmm_type, act=act,
                                                    weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                                                    bias=bias))
            if self.norm:
                self.norms.append(self.norm_type(in_hidden))
        self.convs[-1].last_layer = True

    def forward(self, x, adj, return_mid=False):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        return_mid : bool
            Whether to return hidden representations.
        Returns
        -------
        x : torch.tensor
            Encoded features.
        mid : torch.tensor
            hidden representations. Returned when `return_mid` is True.

        '''
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)
        for i, layer in enumerate(self.convs):
            if self.norm:
                x_res = self.norms[i](x)
                x_res = layer(x_res, adj)
                x = x + x_res
            else:
                x = layer(x,adj)
            if i != self.n_layers - 1:
                mid = x
        if self.output_layer:
            x = self.output_normalization(x)
            x = self.output_linear(x).squeeze(1)
        if return_mid:
            return mid, x.squeeze(1)
        else:
            return x.squeeze(1)


class APPNPEncoder(nn.Module):
    '''
    APPNP encoder from `"Predict then Propagate: Graph Neural Networks meet Personalized PageRank" <https://arxiv.org/abs/1810.05997>`_ paper.

    Parameters
    ----------
    in_channels : int
        Dimensions of input features.
    hidden_channels : int
        Dimensions of hidden representations.
    out_channels : int
        Dimensions of output representations.
    dropout : float
        Dropout rate.
    K : int
        Number of propagations.
    alpha : float
        Teleport probability.
    spmm_type: int
        Specify the multiply funtion used between adj and x.
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1, spmm_type=0):
        super(APPNPEncoder, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.K = K
        self.alpha = alpha
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj, return_mid=False):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        return_mid : bool
            Whether to return hidden representations.
        Returns
        -------
        z : torch.tensor
            Encoded features.
        x : torch.tensor
            hidden representations. Returned when `return_mid` is True.

        '''
        # adj is normalized and sparse
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        z = x
        for i in range(self.K):
            z = (1-self.alpha)*self.spmm(adj,z)+self.alpha*x
        if return_mid:
            return z, z.squeeze(1)
        else:
            return z.squeeze(1)


class GINEncoder(nn.Module):
    '''
    GIN encoder from `"How Powerful are Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Parameters
    ----------
    n_feat : int
        Dimensions of input features.
    n_hidden : int
        Dimensions of hidden representations.
    n_class : int
        Dimensions of output representations.
    n_layers : int
        Number of layers.
    mlp_layers : int
        Number of MLP layers.
    learn_eps : bool
        Whether to set eps as learned parameters.
    spmm_type: int
        Specify the multiply funtion used between adj and x.
    '''
    def __init__(self, n_feat, n_hidden, n_class, n_layers=3, mlp_layers=1, learn_eps=True, spmm_type=0):
        super(GINEncoder, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.n_layers))
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]

        self.mlps = nn.ModuleList()
        if n_layers == 1:
            self.mlps.append(MLPEncoder(n_feat, n_hidden, n_class, mlp_layers, 0))
        else:
            self.mlps.append(MLPEncoder(n_feat, n_hidden, n_hidden, mlp_layers, 0))
            for layer in range(self.n_layers - 2):
                self.mlps.append(MLPEncoder(n_hidden, n_hidden, n_hidden, mlp_layers, 0))
            self.mlps.append(MLPEncoder(n_hidden, n_hidden, n_class, mlp_layers, 0))

    def forward(self, x, adj, return_mid=False):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        return_mid : bool
            Whether to return hidden representations.
        Returns
        -------
        z : torch.tensor
            Encoded features.
        x : torch.tensor
            hidden representations. Returned when `return_mid` is True.

        '''
        if self.learn_eps:
            for i in range(self.n_layers - 1):
                x = self.mlps[i]((1 + self.eps[i]) * x + self.spmm(adj, x))
                x = F.relu(x)
            z = self.mlps[-1]((1 + self.eps[-1]) * x + self.spmm(adj, x))
        else:
            for i in range(self.n_layers - 1):
                x = self.mlps[i](x + self.spmm(adj, x))
                x = F.relu(x)
            z = self.mlps[-1](x + self.spmm(adj, x))

        if return_mid:
            return x, z.squeeze(1)
        else:
            return z.squeeze(1)
