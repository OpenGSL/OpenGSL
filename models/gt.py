'''
This is the GT model from UniMP [https://arxiv.org/pdf/2009.03509.pdf]
implemented by [https://arxiv.org/pdf/2302.11640.pdf] via dgl
'''
import numpy as np
import torch
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax
import torch.nn.functional as F
import scipy.sparse as sp
from utils.utils import get_homophily


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        assert dim % num_heads == 0, 'Dimension mismatch: hidden_dim should be a multiple of num_heads.'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, graph, labels=None, graph_analysis=False):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, values, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.output_linear(x)
        x = self.dropout(x)

        if graph_analysis:
            assert labels is not None, 'error'
            homophily = self.compute_homo(graph, attn_probs, labels)
            return x, homophily
        return x, 0

    def compute_homo(self, graph, attn_weights, labels):
        '''
        Args:
            graph: dgl graph
            attn_weights: [n_edges, n_heads, 1], attention weights learned by a transformer layer

        Returns:
            homophily: [n_heads] the homophily of attention weights of each head

        '''

        n_heads = self.num_heads
        n_nodes = graph.num_nodes()
        homophily = np.zeros(n_heads)
        edges = graph.edges()
        row = edges[0].cpu().numpy()
        col = edges[1].cpu().numpy()
        # values = adj.coalesce().values().numpy()
        # return sp.coo_matrix((values, (row, col)), shape=adj.shape)
        for i in range(n_heads):
            values = attn_weights.squeeze()[:, i].cpu().detach().numpy()
            adj = sp.coo_matrix((values, (row, col)), shape=(n_nodes, n_nodes)).todense()
            homophily[i] = get_homophily(labels, adj)
        return homophily



class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, act):
        super().__init__()
        input_dim = int(dim)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = eval('F.' + act) if not act == 'identity' else lambda x: x
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, n_layers=5, dropout=0.5, input_dropout=0.0, norm_type='LayerNorm', num_heads=8, act='relu', input_layer=True, output_layer=True, ff=True, hidden_dim_multiplier=2):

        super(GT, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.ff = ff
        self.norm_type = eval('nn.' + norm_type)
        self.act = eval('F.' + act) if not act == 'identity' else lambda x: x
        if input_layer:
            self.input_linear = nn.Linear(in_features=nfeat, out_features=nhid)
            self.input_drop = nn.Dropout(input_dropout)
        if output_layer:
            self.output_linear = nn.Linear(in_features=nhid, out_features=nclass)
            self.output_normalization = self.norm_type(nhid)
        self.trans = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        if self.ff:
            self.ffns = nn.ModuleList()
            self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = nfeat
            else:
                in_hidden = nhid
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = nclass
            else:
                out_hidden = nhid
            self.trans.append(TransformerAttentionModule(in_hidden, num_heads, dropout))
            self.norms_1.append(self.norm_type(in_hidden))
            if self.ff:
                self.ffns.append(FeedForwardModule(in_hidden, hidden_dim_multiplier, dropout, act))
                self.norms_2.append(self.norm_type(in_hidden))

    def forward(self, x, graph, labels=None, graph_analysis=False):
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)

        homo_heads = []
        for i, layer in enumerate(self.trans):
            x_res = self.norms_1[i](x)
            x_res, homophily = layer(x_res, graph, labels, graph_analysis)
            x = x + x_res
            if self.ff:
                x_res = self.norms_1[i](x)
                x_res = self.ffns[i](x_res)
                x = x + x_res
            if i == self.n_layers - 1:
                mid = x
            if graph_analysis:
                homo_heads.append(homophily)

        if self.output_layer:
            x = self.output_normalization(x)
            x = self.output_linear(x).squeeze(1)
        return mid, x, homo_heads


if __name__ == '__main__':
    model = GT(1000, 128, 7, 5, 0.5, 0)
    print(model)