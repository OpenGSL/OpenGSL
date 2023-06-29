'''
This is the GT model from UniMP [https://arxiv.org/pdf/2009.03509.pdf]
'''
import numpy as np
import torch
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax
import torch.nn.functional as F
import scipy.sparse as sp
from ...utils.utils import get_homophily
from opengsl.utils.utils import scipy_sparse_to_sparse_tensor


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, dim_out, num_heads, dropout):
        super().__init__()

        assert dim % num_heads == 0, 'Dimension mismatch: hidden_dim should be a multiple of num_heads.'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim_out)
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
            adj = sp.coo_matrix((values, (row, col)), shape=(n_nodes, n_nodes))
            adj = scipy_sparse_to_sparse_tensor(adj)
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

    def __init__(self, nfeat, nhid, nclass, n_layers=5, dropout=0.5, input_dropout=0.0, norm_type='LayerNorm',
                 num_heads=8, act='relu', input_layer=False, output_layer=False, ff=False, hidden_dim_multiplier=2,
                 use_norm=False, use_redisual=False):

        super(GT, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.use_norm = use_norm
        self.use_residual = use_redisual
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
        if self.use_norm:
            self.norms_1 = nn.ModuleList()
        if self.ff:
            self.ffns = nn.ModuleList()
            if self.use_norm:
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
            self.trans.append(TransformerAttentionModule(in_hidden, out_hidden, num_heads, dropout))
            if self.use_norm:
                self.norms_1.append(self.norm_type(in_hidden))
            if self.ff:
                self.ffns.append(FeedForwardModule(in_hidden, hidden_dim_multiplier, dropout, act))
                if self.use_norm:
                    self.norms_2.append(self.norm_type(in_hidden))

    def forward(self, x, graph, labels=None, graph_analysis=False):
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)

        homo_heads = []
        for i, layer in enumerate(self.trans):

            x_res = self.norms_1[i](x) if self.use_norm else x
            x_res, homophily = layer(x_res, graph, labels, graph_analysis)
            x = x + x_res if self.use_residual else x_res

            if self.ff:
                x_res = self.norms_2[i](x) if self.use_norm else x
                x_res = self.ffns[i](x_res)
                x = x + x_res if self.use_residual else x_res
            if i == self.n_layers - 1:
                mid = x
            if graph_analysis:
                homo_heads.append(homophily)

        if self.output_layer:
            if self.use_norm:
                x = self.output_normalization(x)
            x = self.output_linear(x)
        return mid, x.squeeze(1), homo_heads


class GraphTransformerAttn(nn.Module):
    def __init__(self, dim, dim_out, num_heads, concat=True):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.dim_inner = dim_out * num_heads
        self.num_heads = num_heads
        self.concat = concat

        self.attn_query = nn.Linear(in_features=dim, out_features=self.dim_inner)
        self.attn_key = nn.Linear(in_features=dim, out_features=self.dim_inner)
        self.attn_value = nn.Linear(in_features=dim, out_features=self.dim_inner)

    def forward(self, x, graph, labels=None, graph_analysis=False):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.dim_out)
        keys = keys.reshape(-1, self.num_heads, self.dim_out)
        values = values.reshape(-1, self.num_heads, self.dim_out)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.dim_out ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, values, attn_probs)
        if self.concat:
            x = x.reshape(-1, self.dim_inner)
        else:
            x = torch.mean(x, dim=1)

        if graph_analysis:
            assert labels is not None, 'error'
            homophily = self.compute_homo(graph, attn_probs, labels)
            return x, homophily
        return x

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
            adj = sp.coo_matrix((values, (row, col)), shape=(n_nodes, n_nodes))
            adj = scipy_sparse_to_sparse_tensor(adj)
            homophily[i] = get_homophily(labels, adj)
        return homophily


class GatedResidual(nn.Module):
    """ This is the implementation of Eq (5), i.e., gated residual connection between block.
    """
    def __init__(self, dim_in, dim_out, only_gate=False):
        super().__init__()
        self.lin_res = nn.Linear(dim_in, dim_out)
        self.proj = nn.Sequential(
            nn.Linear(dim_out * 3, 1, bias = False),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim_out)
        self.non_lin = nn.ReLU()
        self.only_gate = only_gate

    def forward(self, x, res):
        res = self.lin_res(res)
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input) # Eq (5), this is beta in the paper
        if self.only_gate: # This is for Eq (6), a case when normalizaton and non linearity is not used.
            return x * gate + res * (1 - gate)
        return self.non_lin(self.norm(x * gate + res * (1 - gate)))


class GraphTransformerModel(nn.Module):
    """ This is the overall architecture of the model.
    """

    def __init__(
            self,
            n_feats,
            n_class,
            n_hidden,
            n_layers,
            n_heads=8,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.layers = nn.ModuleList()

        self.input_layer = nn.Linear(n_feats, n_hidden)

        assert n_hidden % n_heads == 0

        for i in range(n_layers):
            if i < n_layers - 1:
                self.layers.append(nn.ModuleList([
                    GraphTransformerAttn(n_hidden, dim_out=int(n_hidden / n_heads), num_heads=n_heads),
                    GatedResidual(n_hidden, n_hidden)
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    GraphTransformerAttn(n_hidden, dim_out=n_class, num_heads=n_heads, concat=False),
                    GatedResidual(n_hidden, n_class, only_gate=True)
                ]))

    def forward(self, input):
        x=input[0]
        graph=input[1]

        x = self.input_layer(x)

        for trans_block in self.layers:
            trans, trans_residual = trans_block
            x = trans_residual(trans(x, graph), x)

        return x


if __name__ == '__main__':
    model = GT(1000, 128, 7, 5, 0.5, 0)
    print(model)