import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn_idgl import GCN
from .gnn_modules import APPNP, GIN


class AnchorGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, with_bias=False, batch_norm=True):
        super(AnchorGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, anchor_mp=True):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=1e-12)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=1e-12)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n_layers=3, dropout=0.5, with_bias=False, batch_norm=True):
        super(AnchorGCN, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.n_layers = n_layers
        self.dropout = dropout
        self.with_bias = with_bias
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        self.layers.append(AnchorGCNLayer(nfeat, nhid, with_bias=with_bias, batch_norm=batch_norm))

        for _ in range(n_layers - 2):
            self.layers.append(AnchorGCNLayer(nhid, nhid, with_bias=with_bias, batch_norm=batch_norm))

        self.layers.append(AnchorGCNLayer(nhid, nclass, with_bias=with_bias, batch_norm=False))


    def forward(self, x, init_adj, cur_node_anchor_adj, graph_skip_conn, first=True, first_init_agg_vec=None,
                init_agg_vec=None, update_adj_ratio=None, dropout=None, first_node_anchor_adj=None):

        if dropout is None:
            dropout = self.dropout

        # layer 1
        first_vec = self.layers[0](x, cur_node_anchor_adj, anchor_mp=True)
        if first:
            init_agg_vec = self.layers[0](x, init_adj, anchor_mp=False)
        else:
            first_vec = update_adj_ratio * first_vec + (1 - update_adj_ratio) * first_init_agg_vec
        node_vec = (1-graph_skip_conn)*first_vec+graph_skip_conn*init_agg_vec
        if self.batch_norm:
            node_vec = self.layers[0].compute_bn(node_vec)
        node_vec = F.dropout(torch.relu(node_vec), dropout, training=self.training)

        # layer 2-n-1
        for encoder in self.layers[1:-1]:

            mid_cur_agg_vec = encoder(node_vec, cur_node_anchor_adj, anchor_mp=True)
            if not first:
                mid_cur_agg_vec = update_adj_ratio*mid_cur_agg_vec+(1-update_adj_ratio)*encoder(node_vec,first_node_anchor_adj,anchor_mp=True)
            node_vec = (1 - graph_skip_conn) * mid_cur_agg_vec + graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False)
            if self.batch_norm:
                node_vec = encoder.compute_bn(node_vec)
            node_vec = F.dropout(torch.relu(node_vec), dropout, training=self.training)

        # layer n
        cur_agg_vec = self.layers[-1](node_vec, cur_node_anchor_adj, anchor_mp=True)
        if not first:
            cur_agg_vec = update_adj_ratio * cur_agg_vec + (1-update_adj_ratio) * self.layers[-1](node_vec, first_node_anchor_adj, anchor_mp=True)
        output = (1 - graph_skip_conn) * cur_agg_vec + graph_skip_conn * self.layers[-1](node_vec, init_adj, anchor_mp=False)
        output = F.log_softmax(output, dim=-1).squeeze(1)
        return first_vec, init_agg_vec, node_vec, output


class GraphLearner(nn.Module):
    def __init__(self, input_size, topk=None, epsilon=None, num_pers=16):
        super(GraphLearner, self).__init__()
        self.topk = topk
        self.epsilon = epsilon

        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, context, anchor=None):
        # return a new adj according to the representation gived

        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        context_fc = context.unsqueeze(0) * expand_weight_tensor  # (num_pers, num_node, dim)
        context_norm = F.normalize(context_fc, p=2, dim=-1)

        if anchor is None:
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)   # (num_node, num_node)
            markoff_value = 0
        else:
            anchors_fc = anchor.unsqueeze(0) * expand_weight_tensor
            anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2)).mean(0)  # (num_node, num_anchor)
            markoff_value = 0

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)
        return attention

    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        device = attention.device
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val).to(device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix


class IDGL(nn.Module):
    def __init__(self, conf, nfeat, nclass):
        super(IDGL, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.hidden_size = conf.model['n_hidden']
        self.dropout = conf.model['dropout']
        self.scalable_run = conf.model['scalable_run'] if 'scalable_run' in conf.model else False
        self.feat_adj_dropout = conf.gsl['feat_adj_dropout']

        if self.scalable_run:
            self.encoder = AnchorGCN(nfeat=nfeat, nhid=conf.model['n_hidden'], nclass=nclass, n_layers=conf.model['n_layers'], dropout=conf.model['dropout'], batch_norm=conf.model['norm'])
        elif conf.model['type'] == 'gcn':
            self.encoder = GCN(nfeat=nfeat, nhid=conf.model['n_hidden'], nclass=nclass,
                                     n_layers=conf.model['n_layers'], dropout=conf.model['dropout'],
                                     batch_norm=conf.model['norm'])
        elif conf.model['type'] == 'appnp':
            self.encoder = APPNP(in_channels=nfeat, hidden_channels=conf.model['n_hidden'], out_channels=nclass, dropout=conf.model['dropout'], K=conf.model['K'], alpha=conf.model['alpha'])
        elif conf.model['type'] == 'gin':
            self.encoder = GIN(n_feat=nfeat, n_hidden=conf.model['n_hidden'], n_class=nclass, n_layers=conf.model['n_layers'], mlp_layers=conf.model['mlp_layers'])
        self.graph_learner = GraphLearner(nfeat,
                                        topk=conf.gsl['graph_learn_topk'],
                                        epsilon=conf.gsl['graph_learn_epsilon'],
                                        num_pers=conf.gsl['graph_learn_num_pers'])

        self.graph_learner2 = GraphLearner(self.nclass if conf.model['type'] == 'appnp' else self.hidden_size,
                                        topk=conf.gsl['graph_learn_topk2'],
                                        epsilon=conf.gsl['graph_learn_epsilon2'],
                                        num_pers=conf.gsl['graph_learn_num_pers'])

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None, anchor_features=None):
        device = node_features.device

        if self.scalable_run:
            node_anchor_adj = graph_learner(node_features, anchor_features)
            return node_anchor_adj

        else:
            raw_adj = graph_learner(node_features)

            assert raw_adj.min().item() >= 0
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=1e-12)   # 归一化

            if graph_skip_conn in (0, None):
                if graph_include_self:
                    adj = adj + torch.eye(adj.size(0)).to(device)
            else:
                adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

            return raw_adj, adj

    def forward(self, node_features, init_adj=None):
        node_features = F.dropout(node_features, self.feat_adj_dropout, training=self.training)
        raw_adj, adj = self.learn_graph(self.graph_learner, node_features, self.graph_skip_conn, init_adj=init_adj)
        adj = F.dropout(adj, self.feat_adj_dropout, training=self.training)
        node_vec = self.encoder(node_features, adj)
        output = F.log_softmax(node_vec, dim=-1).squeeze(1)
        return output, adj


def sample_anchors(node_vec, s):
    idx = torch.randperm(node_vec.size(0))[:s]
    return node_vec[idx], idx


def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=1e-12)
    return diff_


def compute_anchor_adj(node_anchor_adj, anchor_mask=None):
    '''Can be more memory-efficient'''
    anchor_node_adj = node_anchor_adj.transpose(-1, -2)   # (num_anchor, num_node)
    anchor_norm = torch.clamp(anchor_node_adj.sum(dim=-2), min=1e-12) ** -1
    # anchor_adj = torch.matmul(anchor_node_adj, torch.matmul(torch.diag(anchor_norm), node_anchor_adj))
    anchor_adj = torch.matmul(anchor_node_adj, anchor_norm.unsqueeze(-1) * node_anchor_adj)

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

    return anchor_adj