import torch
import torch.nn.functional as F
# from .GCN3 import GraphConvolution, GCN
from .gcn import GCN
from .gnn_modules import APPNP, GIN


class GCNConv_diag(torch.nn.Module):
    '''
    A GCN convolution layer of diagonal matrix multiplication
    '''
    def __init__(self, input_size):
        super(GCNConv_diag, self).__init__()
        self.W = torch.nn.Parameter(torch.ones(input_size))
        # inds = torch.stack([torch.arange(input_size), torch.arange(input_size)]).to(device)
        # self.mW = torch.sparse.FloatTensor(inds, self.W, torch.Size([input_size,input_size]))
        self.input_size = input_size

    def forward(self, input, A):
        hidden = input @ torch.diag(self.W)
        # hidden = torch.sparse.mm(self.mW, input.t()).t()
        output = torch.sparse.mm(A, hidden)
        return output


class GRCN(torch.nn.Module):

    def __init__(self, num_nodes, num_features, num_classes, device, conf):
        super(GRCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        if conf.model['type'] == 'gcn':
            self.conv_task = GCN(num_features, conf.model['n_hidden'], num_classes, conf.model['n_layers'],
                                 conf.model['dropout'], conf.model['input_dropout'], conf.model['norm'],
                                 conf.model['n_linear'], conf.model['spmm_type'], conf.model['act'],
                                 conf.model['input_layer'], conf.model['output_layer'])
        elif conf.model['type']=='appnp':
            self.conv_task = APPNP(num_features, conf.model['n_hidden'], num_classes,
                               dropout=conf.model['dropout'], K=conf.model['K_APPNP'],
                               alpha=conf.model['alpha'], spmm_type=1)
        elif conf.model['type'] == 'gin':
            self.conv_task = GIN(num_features, conf.model['n_hidden'], num_classes,
                               conf.model['n_layers'], conf.model['mlp_layers'], spmm_type=1)
        self.model_type = conf.gsl['model_type']
        if conf.gsl['model_type'] == 'diag':
            self.conv_graph = GCNConv_diag(num_features)
            self.conv_graph2 = GCNConv_diag(num_features)
        else:
            self.conv_graph = GCN(num_features, conf.gsl['n_hidden_1'], conf.gsl['n_hidden_2'], conf.gsl['n_layers'],
                             conf.gsl['dropout'], conf.gsl['input_dropout'], conf.gsl['norm'],
                             conf.gsl['n_linear'], conf.gsl['spmm_type'], conf.gsl['act'],
                             conf.gsl['input_layer'], conf.gsl['output_layer'])

        self.K = conf.gsl['K']
        self.mask = None
        self.Adj_new = None
        self._normalize = conf.gsl['normalize']   # 用来决定是否对node embedding进行normalize
        self.device = device

    def graph_parameters(self):
        if self.model_type == 'diag':
            return list(self.conv_graph.parameters()) + list(self.conv_graph2.parameters())
        else:
            return list(self.conv_graph.parameters())


    def base_parameters(self):
        return list(self.conv_task.parameters())

    def cal_similarity_graph(self, node_embeddings):
        # 一个2head的相似度计算
        # similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
        similarity_graph = torch.mm(node_embeddings[:, :int(self.num_features/2)], node_embeddings[:, :int(self.num_features/2)].t())
        similarity_graph += torch.mm(node_embeddings[:, int(self.num_features/2):], node_embeddings[:, int(self.num_features/2):].t())
        return similarity_graph

    def normalize(self, adj):
        adj = adj.coalesce()
        inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-10)
        D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
        new_values = adj.values() * D_value
        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).to(self.device)

    def _sparse_graph(self, raw_graph, K):
        values, indices = raw_graph.topk(k=int(K), dim=-1)
        assert torch.sum(torch.isnan(values)) == 0
        assert torch.max(indices) < raw_graph.shape[1]

        inds = torch.stack([torch.arange(raw_graph.shape[0]).view(-1,1).expand(-1,int(K)).contiguous().view(1,-1)[0].to(self.device),
                             indices.view(1,-1)[0]])
        inds = torch.cat([inds, torch.stack([inds[1], inds[0]])], dim=1)
        values = torch.cat([values.view(1,-1)[0], values.view(1,-1)[0]])
        return inds, values

    def _node_embeddings(self, input, Adj):
        norm_Adj = self.normalize(Adj)
        if self.model_type == 'diag':
            node_embeddings = torch.tanh(self.conv_graph(input, norm_Adj))
            node_embeddings = self.conv_graph2(node_embeddings, norm_Adj)
        else:
            node_embeddings = self.conv_graph((input, norm_Adj, True))
        if self._normalize:
            node_embeddings = F.normalize(node_embeddings, dim=1, p=2)
        return node_embeddings

    def forward(self, input, Adj):
        Adj.requires_grad = False
        node_embeddings = self._node_embeddings(input, Adj)
        Adj_new = self.cal_similarity_graph(node_embeddings)

        Adj_new_indices, Adj_new_values = self._sparse_graph(Adj_new, self.K)
        Adj_mid = torch.sparse.FloatTensor(Adj_new_indices, Adj_new_values, Adj.size()).to(self.device)
        new_inds = torch.cat([Adj.indices(), Adj_new_indices], dim=1)
        new_values = torch.cat([Adj.values(), Adj_new_values])
        Adj_new = torch.sparse.FloatTensor(new_inds, new_values, Adj.size()).to(self.device)
        Adj_new_norm = self.normalize(Adj_new)

        # x = self.conv1(input, Adj_new_norm)
        # x = F.dropout(F.relu(x), training=self.training, p=self.dropout)
        # x = self.conv2(x, Adj_new_norm)
        _, x = self.conv_task((input, Adj_new_norm, False))

        return x, Adj_mid, Adj_new
