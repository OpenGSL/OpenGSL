from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch_geometric.nn import GCNConv, GATConv
import torch_sparse
# from .gnn_modules import APPNP
from .gcn import GCN

class GraphLearner(nn.Module):
    def __init__(self, input_size, num_pers=16):
        super(GraphLearner, self).__init__()
        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def reset_parameters(self):
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, context):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
        mask = (attention > 0).detach().float()
        attention = attention * mask + 0 * (1 - mask)

        return attention

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, use_bn=False):
        super(MLP, self).__init__()
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
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

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


# class DenseAPPNP(nn.Module):
#     def __init__(self, K, alpha):
#         super().__init__()
#         self.K = K
#         self.alpha = alpha

#     def forward(self, x, adj_t):
#         h = x
#         for k in range(self.K):
#             if adj_t.is_sparse:
#                 x = torch_sparse.spmm(adj_t, x)
#             else:
#                 x = torch.matmul(adj_t, x)
#             x = x * (1 - self.alpha)
#             x += self.alpha * h
#         return x


# class Dense_APPNP_Net(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1):
#         super(Dense_APPNP_Net, self).__init__()
#         self.lin1 = nn.Linear(in_channels, hidden_channels)
#         self.lin2 = nn.Linear(hidden_channels, out_channels)
#         self.prop1 = DenseAPPNP(K, alpha)
#         self.dropout = dropout

#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()

#     def forward(self, x, adj_t):
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin2(x)
#         x = self.prop1(x, adj_t)
#         return x

INF = 1e20
VERY_SMALL_NUMBER = 1e-12

class QModel(nn.Module):
    def __init__(self, graph_skip_conn, nhid, dropout, n_layers, graph_learn_num_pers, d, n, c):
        super(QModel, self).__init__()
        self.graph_skip_conn = graph_skip_conn
        # self.encoder = APPNP(d,nhid,c,dropout,hops,alpha)
        self.encoder = GCN(d, nhid, c, n_layers, dropout)
        # self.encoder = Dense_APPNP_Net(in_channels=d,
        #                                hidden_channels=nhid,
        #                                out_channels=c,
        #                                dropout=dropout,
        #                                K=hops,
        #                                alpha=alpha)

        self.graph_learner1 = GraphLearner(input_size=d, num_pers=graph_learn_num_pers)
        self.graph_learner2 = GraphLearner(input_size=2 * c, num_pers=graph_learn_num_pers)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.graph_learner1.reset_parameters()
        self.graph_learner2.reset_parameters()

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, init_adj=None):
        raw_adj = graph_learner(node_features)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

        return raw_adj, adj

    def forward(self, feats, n_node, edge_index):
        node_features = feats
        train_index = edge_index
        edge_weight = None
        _edge_index, edge_weight = gcn_norm(
            train_index, edge_weight, n_node, False,
            dtype=node_features.dtype)
        row, col = _edge_index
        init_adj_sparse = SparseTensor(row=col, col=row, value=edge_weight,
                                       sparse_sizes=(n_node, n_node))
        init_adj = init_adj_sparse.to_dense()

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, node_features, self.graph_skip_conn, init_adj)
        node_vec_1 = self.encoder([node_features, adj_1, True])

        node_vec_2 = self.encoder([node_features, init_adj, True])
        raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([node_vec_1, node_vec_2], dim=1),
                                            self.graph_skip_conn, init_adj)

        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        adj = 0.5 * adj_1 + 0.5 * adj_2

        return output, adj


class PModel(nn.Module):
    def __init__(self, nhid, dropout, n_layers, graph_learn_num_pers, mlp_layers, no_bn, d, n, c):
        super(PModel, self).__init__()
        # self.encoder1 = APPNP(d,nhid,c,dropout,hops,alpha)
        self.encoder1 = GCN(d, nhid, c, n_layers, dropout)
        # self.encoder1 = Dense_APPNP_Net(in_channels=d,
        #                                 hidden_channels=nhid,
        #                                 out_channels=c,
        #                                 dropout=dropout,
        #                                 K=hops,
        #                                 alpha=alpha)

        self.encoder2 = MLP(in_channels=d,
                            hidden_channels=nhid,
                            out_channels=c,
                            num_layers=mlp_layers,
                            dropout=dropout,
                            use_bn=not no_bn)

        self.graph_learner1 = GraphLearner(input_size=d, num_pers=graph_learn_num_pers)
        self.graph_learner2 = GraphLearner(input_size=2 * c, num_pers=graph_learn_num_pers)

    def reset_parameters(self):
        self.encoder1.reset_parameters()
        self.encoder2.reset_parameters()
        self.graph_learner.reset_parameters()
        self.graph_learner2.reset_parameters()

    def learn_graph(self, graph_learner, node_features):
        raw_adj = graph_learner(node_features)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

        return raw_adj, adj

    def forward(self, feats):
        node_features = feats

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, node_features)
        node_vec_1 = self.encoder1([node_features, adj_1, True])

        node_vec_2 = self.encoder2(node_features)
        raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([node_vec_1, node_vec_2], dim=1))

        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        adj = 0.5 * adj_1 + 0.5 * adj_2

        return output, adj


class WSGNN(nn.Module):
    def __init__(self, graph_skip_conn, nhid, dropout, n_layers, graph_learn_num_pers, mlp_layers, no_bn, d, n, c):
        super(WSGNN, self).__init__()
        self.P_Model = PModel(nhid, dropout, n_layers, graph_learn_num_pers, mlp_layers, no_bn, d, n, c)
        self.Q_Model = QModel(graph_skip_conn, nhid, dropout, n_layers, graph_learn_num_pers, d, n, c)

    def reset_parameters(self):
        self.P_Model.reset_parameters()
        self.Q_Model.reset_parameters()

    def forward(self, feats, n_node, edge_index):
        q_y, q_a = self.Q_Model.forward(feats, n_node, edge_index)
        p_y, p_a = self.P_Model.forward(feats)
        return p_y, p_a, q_y, q_a

class ELBONCLoss(nn.Module):
    def __init__(self):
        super(ELBONCLoss, self).__init__()

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
        loss_p_y = F.nll_loss(log_p_y_obs, y_obs) - torch.mean(q_y_miss * log_p_y_miss)
        loss_q_y = torch.mean(q_y_miss * log_q_y_miss)

        loss_y_obs = 10 * F.nll_loss(log_q_y_obs, y_obs)

        loss = loss_p_y + loss_q_y + loss_y_obs

        return loss