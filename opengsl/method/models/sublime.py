import dgl
import torch
import torch.nn as nn
from opengsl.method.models.gcn import GCN
from opengsl.method.models.gnn_modules import APPNP, GIN
import dgl.function as fn
import numpy as np
import torch.nn.functional as F
import copy
EOS = 1e-10


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda(), samples


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph


class GCNConv_dgl(nn.Module):
    # to be changed to pyg in future versions
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, sparse, conf=None):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            if conf.model['type']=='gcn':
                self.model = GCN(nfeat=in_dim, nhid=hidden_dim, nclass=emb_dim, n_layers=nlayers, dropout=dropout,
                                 input_layer=False, output_layer=False, spmm_type=0)
            elif conf.model['type']=='appnp':
                self.model = APPNP(in_dim, hidden_dim, emb_dim,
                                    dropout=dropout, K=conf.model['K'],
                                    alpha=conf.model['alpha'])
            elif conf.model['type'] == 'gin':
                self.model = GIN(in_dim, hidden_dim, emb_dim,
                               nlayers, conf.model['mlp_layers'])
        self.proj_head = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU(inplace=True),
                                           nn.Linear(proj_dim, proj_dim))

    def forward(self, x, Adj_):

        if self.sparse:
            for conv in self.gnn_encoder_layers[:-1]:
                x = conv(x, Adj_)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gnn_encoder_layers[-1](x, Adj_)
        else:
            x = self.model((x, Adj_, True))
        z = self.proj_head(x)
        return z, x


class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse, conf=None):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, sparse, conf)
        self.dropout_adj = dropout_adj
        self.sparse = sparse

    def forward(self, x, Adj_, branch=None):

        # edge dropping
        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj, training=self.training)
        else:
            Adj = F.dropout(Adj_, p=self.dropout_adj, training=self.training)

        # get representations
        z, embedding = self.encoder(x, Adj)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)   # 计算的是cos相似度
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss


class GCN_SUB(nn.Module):
    # to be changed to pyg in future versions
    def __init__(self, nfeat, nhid, nclass, n_layers=5, dropout=0.5, dropout_adj=0.5, sparse=0):
        super(GCN_SUB, self).__init__()
        self.layers = nn.ModuleList()
        self.sparse = sparse
        self.dropout_adj_p = dropout_adj
        self.dropout = dropout

        if sparse:
            self.layers.append(GCNConv_dgl(nfeat, nhid))
            for _ in range(n_layers - 2):
                self.layers.append(GCNConv_dgl(nhid, nhid))
            self.layers.append(GCNConv_dgl(nhid, nclass))
        else:
            self.model = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass, n_layers=n_layers, dropout=dropout,
                             input_layer=False, output_layer=False, spmm_type=0)

    def forward(self, x, Adj):

        if self.sparse:
            Adj = copy.deepcopy(Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = F.dropout(Adj, p=self.dropout_adj_p, training=self.training)

        if self.sparse:
            for i, conv in enumerate(self.layers[:-1]):
                x = conv(x, Adj)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[-1](x, Adj)
            return x.squeeze(1)
        else:
            return self.model((x, Adj, True))


if __name__ == '__main__':
    pass




