from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch_geometric.nn import GCNConv, GATConv
import torch_sparse
from .gnn_modules import APPNP, GIN
from .gcn import GCN
from opengsl.method.metric import WeightedCosine
from opengsl.method.models.gnn_modules import MLP
from opengsl.method.functional import normalize


INF = 1e20
VERY_SMALL_NUMBER = 1e-12


class QModel(nn.Module):
    def __init__(self, graph_skip_conn, nhid, dropout, n_layers, graph_learn_num_pers, d, n, c, conf):
        super(QModel, self).__init__()
        self.graph_skip_conn = graph_skip_conn
        if conf.model['type'] == 'gcn':
            self.encoder = GCN(d, nhid, c, n_layers, dropout)
        elif conf.model['type'] == 'appnp':
            self.encoder = APPNP(d, nhid, c, dropout, conf.model['hops'], conf.model['alpha'])
        elif conf.model['type'] == 'gin':
            self.encoder = GIN(d,nhid,c,n_layers,conf.model['mlp_gin'])

        self.graph_learner1 = WeightedCosine(d_in=d, num_pers=graph_learn_num_pers)
        self.graph_learner2 = WeightedCosine(d_in=2 * c, num_pers=graph_learn_num_pers)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.graph_learner1.reset_parameters()
        self.graph_learner2.reset_parameters()

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, init_adj=None):
        raw_adj = graph_learner(node_features, True)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj
        return raw_adj, adj

    def forward(self, feats, adj):
        node_features = feats
        init_adj = normalize(adj)

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, node_features, self.graph_skip_conn, init_adj)
        node_vec_1 = self.encoder([node_features, adj_1, True])

        node_vec_2 = self.encoder([node_features, init_adj, True])
        if len(node_vec_2.shape) == 2:
            raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([node_vec_1, node_vec_2], dim=1),
                                                self.graph_skip_conn, init_adj)
        else:
            raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2,
                                                torch.stack([node_vec_1, node_vec_2]).transpose(0, 1),
                                                self.graph_skip_conn, init_adj)
        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        adj = 0.5 * adj_1 + 0.5 * adj_2

        return output, adj


class PModel(nn.Module):
    def __init__(self, nhid, dropout, n_layers, graph_learn_num_pers, mlp_layers, no_bn, d, n, c, conf):
        super(PModel, self).__init__()
        if conf.model['type'] == 'gcn':
            self.encoder1 = GCN(d, nhid, c, n_layers, dropout)
        elif conf.model['type'] == 'appnp':
            self.encoder1 = APPNP(d, nhid, c, dropout, conf.model['hops'], conf.model['alpha'])
        elif conf.model['type'] == 'gin':
            self.encoder1 = GIN(d, nhid, c, n_layers, conf.model['mlp_gin'])

        self.encoder2 = MLP(in_channels=d,
                            hidden_channels=nhid,
                            out_channels=c,
                            num_layers=mlp_layers,
                            dropout=dropout,
                            use_bn=not no_bn)

        self.graph_learner1 = WeightedCosine(d_in=d, num_pers=graph_learn_num_pers)
        self.graph_learner2 = WeightedCosine(d_in=2 * c, num_pers=graph_learn_num_pers)

    def reset_parameters(self):
        self.encoder1.reset_parameters()
        self.encoder2.reset_parameters()
        self.graph_learner.reset_parameters()
        self.graph_learner2.reset_parameters()

    def learn_graph(self, graph_learner, node_features):
        raw_adj = graph_learner(node_features, True)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        return raw_adj, adj

    def forward(self, feats):
        node_features = feats

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, node_features)
        node_vec_1 = self.encoder1([node_features, adj_1, True])

        node_vec_2 = self.encoder2(node_features).squeeze(1)
        if len(node_vec_2.shape) == 2:
            raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([node_vec_1, node_vec_2], dim=1))
        else:
            raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.stack([node_vec_1, node_vec_2]).transpose(0,1))
        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        adj = 0.5 * adj_1 + 0.5 * adj_2

        return output, adj


class WSGNN(nn.Module):
    def __init__(self, graph_skip_conn, nhid, dropout, n_layers, graph_learn_num_pers, mlp_layers, no_bn, d, n, c, conf):
        super(WSGNN, self).__init__()
        self.P_Model = PModel(nhid, dropout, n_layers, graph_learn_num_pers, mlp_layers, no_bn, d, n, c, conf)
        self.Q_Model = QModel(graph_skip_conn, nhid, dropout, n_layers, graph_learn_num_pers, d, n, c, conf)

    def reset_parameters(self):
        self.P_Model.reset_parameters()
        self.Q_Model.reset_parameters()

    def forward(self, feats, adj):
        q_y, q_a = self.Q_Model.forward(feats, adj)
        p_y, p_a = self.P_Model.forward(feats)
        return p_y, p_a, q_y, q_a


class ELBONCLoss(nn.Module):
    def __init__(self, binary=False):
        super(ELBONCLoss, self).__init__()
        self.binary = binary

    def forward(self, labels, train_mask, log_p_y, log_q_y):
        y_obs = labels[train_mask]
        log_p_y_obs = log_p_y[train_mask]
        p_y_obs = torch.exp(log_p_y_obs)
        log_p_y_miss = log_p_y[train_mask == 0]
        p_y_miss = torch.exp(log_p_y_miss)
        log_q_y_obs = log_q_y[train_mask]
        q_y_obs = torch.exp(log_q_y_obs)
        log_q_y_miss = log_q_y[train_mask == 0]
        q_y_miss = torch.exp(log_q_y_miss)
        if self.binary:
            loss_p_y = F.binary_cross_entropy_with_logits(log_p_y_obs, y_obs) - torch.mean(
                torch.sigmoid(log_q_y_miss) * F.logsigmoid(log_p_y_miss))
            loss_q_y = torch.mean(torch.sigmoid(log_q_y_miss) * F.logsigmoid(log_q_y_miss))
            loss_y_obs = 10 * F.binary_cross_entropy_with_logits(log_q_y_obs, y_obs)
        else:
            loss_p_y = F.nll_loss(log_p_y_obs, y_obs) - torch.mean(q_y_miss * log_p_y_miss)
            loss_q_y = torch.mean(q_y_miss * log_q_y_miss)
            loss_y_obs = 10 * F.nll_loss(log_q_y_obs, y_obs)

        loss = loss_p_y + loss_q_y + loss_y_obs

        return loss