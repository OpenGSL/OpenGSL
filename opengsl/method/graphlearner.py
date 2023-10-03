import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits
import torch.nn.init as init
from sklearn.neighbors import kneighbors_graph
import numpy as np
from opengsl.method.encoder import AttentiveEncoder, MLPEncoder
from opengsl.method.metric import WeightedCosine, Cosine
from opengsl.method.transform import KNN, NonLinear
from opengsl.method.functional import knn_fast
import dgl


class GraphLearner(nn.Module):

    def __init__(self, encoder=None, metric=None, postprocess=None):
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

    def forward(self, x, adj):
        x = self.encoder(x, adj)
        adj = self.metric(x)
        for p in self.postprocess:
            adj = p(adj)
        return adj


class OneLayerNN(GraphLearner):

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


class FGPLearner(GraphLearner):
    def __init__(self, n):
        super(FGPLearner, self).__init__()
        self.Adj = nn.Parameter(torch.FloatTensor(n, n))

    def reset_parameters(self, features, k, metric, i):
        adj = kneighbors_graph(features, k, metric=metric)
        adj = np.array(adj.todense(), dtype=np.float32)
        adj += np.eye(adj.shape[0])
        adj = adj * i - i
        self.Adj.data.copy_(torch.tensor(adj))

    def forward(self, h):
        Adj = F.elu(self.Adj) + 1
        return Adj


class AttLearner(GraphLearner):

    def __init__(self, nlayers, isize, k, i, sparse, act):
        encoder = AttentiveEncoder(nlayers, isize, act)
        metric = WeightedCosine(isize, 1, False)
        knn = KNN(k + 1)
        self.non_linear = NonLinear('relu', i)
        postprocess = [knn, self.non_linear]

        super(AttLearner, self).__init__(encoder, metric, postprocess)
        self.k = k
        self.sparse = sparse


    def forward(self, x):
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

    def __init__(self, nlayers, isize, k, i, sparse, act):
        encoder = MLPEncoder(isize, isize, isize, nlayers, 0, False, act)
        encoder.param_init()
        metric = Cosine()
        knn = KNN(k + 1)
        self.non_linear = NonLinear('relu', i)
        postprocess = [knn, self.non_linear]
        super(MLPLearner, self).__init__(encoder, metric, postprocess)
        self.k = k
        self.sparse = sparse


    def forward(self, x):
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
                adj = p(adj)
            return adj