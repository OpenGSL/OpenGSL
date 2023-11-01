import torch
from opengsl.module.functional import normalize
from opengsl.module.encoder import GCNDiagEncoder, GCNEncoder, APPNPEncoder, GINEncoder
from opengsl.module.fuse import Interpolate
from opengsl.module.transform import Normalize, KNN, Symmetry
from opengsl.module.metric import InnerProduct


class GRCN(torch.nn.Module):

    def __init__(self, num_nodes, num_features, num_classes, device, conf):
        super(GRCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        if conf.model['type'] == 'gcn':
            self.conv_task = GCNEncoder(num_features, conf.model['n_hidden'], num_classes, conf.model['n_layers'],
                                 conf.model['dropout'], conf.model['input_dropout'], conf.model['norm'],
                                 conf.model['n_linear'], conf.model['spmm_type'], conf.model['act'],
                                 conf.model['input_layer'], conf.model['output_layer'])
        elif conf.model['type'] == 'appnp':
            self.conv_task = APPNPEncoder(num_features, conf.model['n_hidden'], num_classes,
                               dropout=conf.model['dropout'], K=conf.model['K_APPNP'],
                               alpha=conf.model['alpha'], spmm_type=1)
        elif conf.model['type'] == 'gin':
            self.conv_task = GINEncoder(num_features, conf.model['n_hidden'], num_classes,
                               conf.model['n_layers'], conf.model['mlp_layers'], spmm_type=1)
        self.model_type = conf.gsl['model_type']
        if conf.gsl['model_type'] == 'diag':
            self.conv_graph = GCNDiagEncoder(2, num_features)
        else:
            self.conv_graph = GCNEncoder(num_features, conf.gsl['n_hidden_1'], conf.gsl['n_hidden_2'], conf.gsl['n_layers'],
                             conf.gsl['dropout'], conf.gsl['input_dropout'], conf.gsl['norm'],
                             conf.gsl['n_linear'], conf.gsl['spmm_type'], conf.gsl['act'],
                             conf.gsl['input_layer'], conf.gsl['output_layer'])

        self.K = conf.gsl['K']
        self._normalize = conf.gsl['normalize']   # 用来决定是否对node embedding进行normalize

        self.metric = InnerProduct()
        self.normalize_a = Normalize(add_loop=False)
        self.normalize_e = Normalize('row-norm', p=2)
        self.knn = KNN(self.K, sparse_out=True)
        self.sym = Symmetry(1)
        self.fuse = Interpolate(1, 1)

    def graph_parameters(self):
        return list(self.conv_graph.parameters())

    def base_parameters(self):
        return list(self.conv_task.parameters())

    def cal_similarity_graph(self, node_embeddings):
        # 一个2head的相似度计算
        # 完全等价于普通cosine
        similarity_graph = self.metric(node_embeddings[:, :int(self.num_features/2)])
        similarity_graph += self.metric(node_embeddings[:, int(self.num_features/2):])
        return similarity_graph

    def _sparse_graph(self, raw_graph):
        new_adj = self.knn(adj=raw_graph)
        new_adj = self.sym(new_adj)
        return new_adj

    def _node_embeddings(self, input, Adj):
        norm_Adj = self.normalize_a(Adj)
        node_embeddings = self.conv_graph(input, norm_Adj)
        if self._normalize:
            node_embeddings = self.normalize_e(node_embeddings)
        return node_embeddings

    def forward(self, input, Adj):
        adjs = {}
        Adj.requires_grad = False
        node_embeddings = self._node_embeddings(input, Adj)
        Adj_new = self.cal_similarity_graph(node_embeddings)
        Adj_new = self._sparse_graph(Adj_new)
        Adj_final = self.fuse(Adj_new, Adj)
        Adj_final_norm = self.normalize_a(Adj_final.coalesce())
        x = self.conv_task(input, Adj_final_norm)

        adjs['new'] = Adj_new
        adjs['final'] = Adj_final

        return x, adjs
