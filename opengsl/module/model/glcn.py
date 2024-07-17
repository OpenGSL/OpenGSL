import torch
from opengsl.module.encoder import GNNEncoder_OpenGSL
from opengsl.module.graphlearner import OneLayerNN
from opengsl.module.regularizer import norm_regularizer, smoothness_regularizer
from torch_sparse import SparseTensor


class GLCN(torch.nn.Module):

    def __init__(self, n_feat, n_classes, conf):
        super(GLCN, self).__init__()
        if conf.model['type'] == 'gcn':
            self.gnn_encoder = GNNEncoder_OpenGSL(n_feat, n_classes, **conf.model)
        self.graph_learner = OneLayerNN(n_feat, conf.model['n_hidden_graph'], p_dropout=conf.model['dropout'])
        self.loss_lamb1 = conf.training['loss_lamb1']
        self.loss_lamb2 = conf.training['loss_lamb2']

    def forward(self, x, adj):
        adjs = {}
        others = {}
        edge = adj.indices()
        new_adj, new_x = self.graph_learner(x, edge)
        adjs['final'] = new_adj
        z = self.gnn_encoder(x, SparseTensor.from_torch_sparse_coo_tensor(new_adj))

        # calculate some loss items
        loss1 = smoothness_regularizer(new_x, new_adj, symmetric=True)
        loss2 = norm_regularizer(new_adj)
        others['loss'] = self.loss_lamb1 * loss1 + self.loss_lamb2 * loss2
        return z, adjs, others

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
