import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits
import torch.nn.init as init
from sklearn.neighbors import kneighbors_graph
import numpy as np
from opengsl.module.encoder import AttentiveEncoder, MLPEncoder, GCNEncoder
from opengsl.module.metric import WeightedCosine, Cosine
from opengsl.module.transform import KNN, NonLinear
from opengsl.module.functional import knn_fast
import dgl


class GraphLearner(nn.Module):
    '''
    Base Graph Learner to learn new structure from features and original structure. It defines the abstract learning procedures
    including 4 components.

    Parameters
    ----------
    encoder : nn.Module
        Encoder to generate node embeddings. See `encoder` for more details.
    metric : nn.Module
        Metric to generate pairwise similarities. See `metric` for more details.
    postprocess : [object, ]
        List contraining postprocess objects in `transform`.
    fuse : object
        Fuse object to fuse the learned structure and original structure. See `fuse` for more details.
    '''
    def __init__(self, encoder=None, metric=None, postprocess=None, fuse=None):
        super(GraphLearner, self).__init__()
        self.encoder = lambda x, adj: x
        if encoder:
            self.encoder = encoder
        self.postprocess = [lambda x, adj: adj]
        if postprocess:
            self.postprocess = postprocess
        self.metric = lambda x, adj: x @ x.T
        if metric:
            self.metric = metric
        self.fuse = lambda adj, raw_adj: adj
        if fuse:
            self.fuse = fuse
    def forward(self, x, input_adj, raw_adj=None, return_mid=False):
        '''
        Function to learn new structure from features and original structure.

        Parameters
        ----------
        x : torch.tensor
            Features.
        input_adj : torch.tensor
            Input structure.
        raw_adj : torch.tensor
            Original structure. `input_adj` will be used if set to ``None``.
        return_mid : bool
            Whether to return structure before fusing.

        Returns
        -------
        final_adj : torch.tensor
            Learned structure.
        adj : torch.tensor
            Learned structure before fusing.

        '''
        x = self.encoder(x, input_adj)
        adj = self.metric(x)
        for p in self.postprocess:
            adj = p(adj=adj)
        if raw_adj is None:
            raw_adj = input_adj
        final_adj = self.fuse(adj, raw_adj)
        if return_mid:
            return final_adj, adj
        else:
            return final_adj


class OneLayerNN(GraphLearner):
    '''
    Graph Learner adapted from `"Semi-supervised Learning with
    Graph Learning-Convolutional Networks" <https://ieeexplore.ieee.org/document/8953909/authors#authors>`_ paper.

    Parameters
    ----------
    d_in : int
        Input dimensions.
    d_out : int
        Output dimensions.
    p_dropout : float
        Dropout rate.
    '''

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
        '''
        Parameters
        ----------
        x : torch.tensor
            Features.
        edge : torch.tensor
            Structure stored in edge list form.
        return_h : bool
            Whether to return hidden representations.

        Returns
        -------
        adj : torch.tensor
            Learned structure.
        h : torch.tensor
            Hidden representations.

        '''
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


class AttLearner(GraphLearner):
    '''
    Graph Learner adapted from `"Towards Unsupervised Deep Graph Structure Learning" <https://arxiv.org/abs/2201.06367>`_ paper.

    Parameters
    ----------
    n_layers : int
        Number of layers of attentive encoder.
    isize : int
        Dimensions of input features.
    k : int
        K used in KNN algorithm.
    i : int
        Integer used in nonlinear function.
    sparse: bool
        Whether to use sparse versions.
    act : str
        Specify the activation function used in attentive encoder.
    '''

    def __init__(self, n_layers, isize, k, i, sparse, act):
        encoder = AttentiveEncoder(n_layers, isize, act)
        metric = WeightedCosine(isize, 1, False)
        knn = KNN(k + 1)
        non_linear = NonLinear('relu', i)
        postprocess = [knn, non_linear]
        super(AttLearner, self).__init__(encoder, metric, postprocess)
        self.non_linear = non_linear
        self.k = k
        self.sparse = sparse

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.tensor
            Features.

        Returns
        -------
        adj : torch.tensor
            Learned structure.

        '''
        if self.sparse:
            # To be integrated in future versions
            embeddings = self.encoder(x, None)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = self.non_linear(values_)
            adj = dgl.graph((rows_, cols_), num_nodes=x.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            x = self.encoder(x, None)
            adj = self.metric(x)
            for p in self.postprocess:
                adj = p(adj)
            return adj


class MLPLearner(GraphLearner):
    '''
    Graph Learner adapted from `"Towards Unsupervised Deep Graph Structure Learning" <https://arxiv.org/abs/2201.06367>`_ paper.

    Parameters
    ----------
    n_layers : int
        Number of layers of mlp encoder.
    isize : int
        Dimensions of input features.
    k : int
        K used in KNN algorithm.
    i : int
        Integer used in nonlinear function.
    sparse: bool
        Whether to use sparse versions.
    act : str
        Specify the activation function used in mlp encoder.
    '''

    def __init__(self, n_layers, isize, k, i, sparse, act):
        encoder = MLPEncoder(isize, isize, isize, n_layers, 0, False, act)
        encoder.param_init()
        metric = Cosine()
        knn = KNN(k + 1)
        non_linear = NonLinear('relu', i)
        postprocess = [knn, non_linear]
        super(MLPLearner, self).__init__(encoder, metric, postprocess)
        self.k = k
        self.sparse = sparse
        self.non_linear = non_linear

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.tensor
            Features.

        Returns
        -------
        adj : torch.tensor
            Learned structure.

        '''
        if self.sparse:
            # To be integrated in future versions
            embeddings = self.internal_forward(x)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = self.non_linear(values_)
            adj = dgl.graph((rows_, cols_), num_nodes=x.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            x = self.encoder(x, None)
            adj = self.metric(x)
            for p in self.postprocess:
                adj = p(adj=adj)
            return adj


class GLNLearner(GraphLearner):
    '''
    Graph Learner adapted from `"How Attentive are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>`_ paper.
    '''
    def __init__(self, d_in, d_hidden, n):
        '''
        Parameters
        ----------
        d_in
        d_hidden
        n
        '''
        super(GLNLearner, self).__init__()
        self.encoder = GCNEncoder(d_in, d_hidden, d_hidden, n_layers=2, dropout=0)
        self.Z = nn.Parameter(torch.FloatTensor(d_hidden, d_hidden))
        self.M = nn.Parameter(torch.FloatTensor(n, n))
        self.Q = nn.Parameter(torch.FloatTensor(d_hidden, d_hidden))

    def forward(self, x, adj, K=5, return_x=False):
        '''

        Parameters
        ----------
        x
        adj
        K
        return_x

        Returns
        -------

        '''
        for i in range(K):
            x = self.encoder(x, adj)
            adj = self.M @ x @ self.Q @ F.relu(x @ self.Z) @ self.M.T
            adj = F.sigmoid(adj)
        if return_x:
            return x, adj
        else:
            return adj