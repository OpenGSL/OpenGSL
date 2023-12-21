import torch
import torch.nn as nn
import torch.nn.functional as F
from opengsl.module.encoder import GCNEncoder


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, n_nodes, n_class, n_anchors, topk=None, epsilon=None, n_pers=16):
        super(GraphLearner, self).__init__()
        self.n_nodes = n_nodes
        self.n_class = n_class
        self.n_anchors = n_anchors
        self.topk = topk
        self.epsilon = epsilon

        self.weight_tensor = torch.Tensor(n_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

        self.weight_tensor_for_pe = torch.Tensor(self.n_anchors, hidden_size)
        self.weight_tensor_for_pe = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_for_pe))


    def forward(self, context, position_encoding, gpr_rank, position_flag, ctx_mask=None):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)

        attention = torch.bmm(context_norm, context_norm.transpose(-1, -2)).mean(0)

        if position_flag == 1:
            pe_fc = torch.mm(position_encoding, self.weight_tensor_for_pe)
            pe_attention = torch.mm(pe_fc, pe_fc.transpose(-1, -2))
            attention = (attention * 0.5 + pe_attention * 0.5) * gpr_rank
        else:
            attention = attention * gpr_rank

        markoff_value = 0

        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            if not self.epsilon == 0:
                attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention


    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val).cuda()
        return weighted_adjacency_matrix


    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()

        try:
            weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        except:
            attention_np = attention.cpu().detach().numpy()
            mask_np = mask.cpu().detach().numpy()
            weighted_adjacency_matrix_np = attention_np * mask_np + markoff_value * (1 - mask_np)
            weighted_adjacency_matrix = torch.from_numpy(weighted_adjacency_matrix_np).cuda()

        return weighted_adjacency_matrix



class PASTEL(nn.Module):
    def __init__(self, n_nodes, n_feat, nclass, config, position_encoding, gpr_rank):
        super(PASTEL, self).__init__()
        
        self.graph_skip_conn = config.model['graph_skip_conn']
        self.config = config

        self.position_encoding = position_encoding
        self.gpr_rank = gpr_rank
        self.position_flag = 0
        self.graph_learner = GraphLearner(n_feat,
                                config.model['graph_learn_hidden_size'],
                                n_nodes=n_nodes, n_class=nclass,
                                n_anchors=config.model['num_anchors'],
                                topk=config.model['graph_learn_topk'],
                                epsilon=config.model['graph_learn_epsilon'],
                                n_pers=config.model['graph_learn_num_pers'])

        self.backbone = GCNEncoder(nfeat=n_feat, nhid=config.model['n_hidden'], nclass=nclass, n_layers=config.model['n_layers'], 
                                   dropout=config.model['dropout'], input_layer=False, output_layer=False, spmm_type=0).cuda()
        
        
    def learn_graph(self, graph_learner, node_features, position_encoding=None, gpr_rank=None, position_flag=False, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        raw_adj = graph_learner(node_features, position_encoding, gpr_rank, position_flag)
        assert raw_adj.min().item() >= 0
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=1e-12)

        if graph_skip_conn in (0, None):
            if graph_include_self:
                adj = adj + torch.eye(adj.size(0)).cuda()
        else:
            adj = (1 - graph_skip_conn) * adj + graph_skip_conn * init_adj
        return raw_adj, adj


    def forward(self, node_features, init_adj=None):
        node_features = F.dropout(node_features, self.config.model.get('feat_adj_dropout', 0), training=self.training)
        raw_adj, adj = self.learn_graph(self.graph_learner, node_features, self.position_encoding, self.gpr_rank, self.position_flag ,graph_skip_conn=self.graph_skip_conn, init_adj=init_adj)
        adj = F.dropout(adj, self.config.model.get('feat_adj_dropout', 0), training=self.training)
        node_vec = self.backbone(node_features, adj)
        output = F.log_softmax(node_vec, dim=-1)
        return output, raw_adj, adj