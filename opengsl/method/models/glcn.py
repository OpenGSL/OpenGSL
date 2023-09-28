import torch
import torch.nn.functional as F
from .gcn import GCN
from opengsl.method.metric import OneLayerNN
from opengsl.method.regularizer import norm_regularizer, smoothness_regularizer, smoothness_regularizer_direct


class GLCN(torch.nn.Module):

    def __init__(self, n_feat, n_classes, conf):
        super(GLCN, self).__init__()
        if conf.model['type'] == 'gcn':
            self.gnn_encoder = GCN(n_feat, conf.model['n_hidden'], n_classes, conf.model['n_layers'],
                                 conf.model['dropout'], conf.model['input_dropout'], conf.model['norm'],
                                 conf.model['n_linear'], conf.model['spmm_type'], conf.model['act'],
                                 conf.model['input_layer'], conf.model['output_layer'])
        self.graph_learner = OneLayerNN(n_feat, conf.model['n_hidden_graph'], p_dropout=conf.model['dropout'])
        self.loss_lamb1 = conf.training['loss_lamb1']
        self.loss_lamb2 = conf.training['loss_lamb2']

    def forward(self, x, adj):
        adjs = {}
        others = {}
        edge = adj.indices()
        new_adj, new_x = self.graph_learner(x, edge)
        adjs['final'] = new_adj
        z = self.gnn_encoder([x, new_adj])

        # calculate some loss items
        loss1 = smoothness_regularizer(new_x, new_adj, symmetric=True)
        loss2 = norm_regularizer(new_adj, p='norm')
        others['loss'] = self.loss_lamb1 * loss1 + self.loss_lamb2 * loss2
        return z, adjs, others
