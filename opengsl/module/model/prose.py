import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn import metrics
import torch.nn as nn
import copy
from torch.nn import Sequential, Linear, ReLU
from opengsl.module.functional import normalize, knn, enn, symmetry
from opengsl.module.metric import Cosine
from opengsl.module.encoder import GCNEncoder, MLPEncoder

EOS = 1e-10


class GCNConv_dense(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNConv_dense, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class Stage_GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, osize, k, knn_metric, sparse, act, internal_type, ks, share_up_gnn, fusion_ratio,
                 stage_fusion_ratio,
                 epsilon, add_vertical_position, v_pos_dim, dropout_v_pos, up_gnn_nlayers, dropout_up_gnn,
                 add_embedding):
        super(Stage_GNN_learner, self).__init__()

        self.internal_layers = nn.ModuleList()
        print('Stage Internal type =', internal_type)
        self.internal_type = internal_type
        if self.internal_type == 'gnn':
            self.encoder1 = GCNEncoder(n_feat=isize, nhid=osize, n_class=osize, n_layers=nlayers, dropout=0,
                                       input_layer=False, output_layer=False, spmm_type=0, act=act)
        elif self.internal_type == 'mlp':
            self.encoder1 = MLPEncoder(in_channels=isize, hidden_channels=osize, out_channels=osize, n_layers=nlayers, dropout=0,
                                    activation=act, use_bn=False)

        self.k = k
        self.sparse = sparse
        self.act = act
        self.epsilon = epsilon
        self.fusion_ratio = fusion_ratio
        self.add_embedding = add_embedding
        ## stage module
        self.ks = ks
        self.l_n = len(self.ks)
        self.metric = Cosine()

        if self.l_n > 0:
            self.stage_fusion_ratio = stage_fusion_ratio
            # down score
            self.score_layer = GCNConv_dense(osize, 1)
            self.share_up_gnn = share_up_gnn
            self.up_gnn_nlayers = up_gnn_nlayers
            if self.up_gnn_nlayers > 1:
                self.dropout_up_gnn = dropout_up_gnn

            self.up_gnn_layers = nn.ModuleList()
            if self.share_up_gnn:
                self.encoder2 = GCNEncoder(n_feat=osize, nhid=osize, n_class=osize, n_layers=nlayers, dropout=self.dropout_up_gnn,
                                           input_layer=False, output_layer=False, spmm_type=0, act='F.relu')
            else:
                pass

            self.add_vertical_position = add_vertical_position
            if self.add_vertical_position:
                self.dropout_v_pos = dropout_v_pos
                self.v_pos_dim = v_pos_dim
                self.vertival_pos_embedding = nn.Embedding(
                    self.l_n + 1, self.v_pos_dim)
                self.map_v_pos_linear1 = nn.Linear(osize + self.v_pos_dim, osize)
                self.map_v_pos_linear2 = nn.Linear(osize, osize)

    def forward(self, features, adj):

        embeddings = self.encoder1(features, adj)
        cur_embeddings = embeddings
        adj_ = adj
        embeddings_ = embeddings

        if self.l_n > 0:
            indices_list = []
            down_outs = []
            n_node = features.shape[0]
            pre_idx = torch.arange(0, n_node).long()
            for i in range(self.l_n):  # [0,1,2]
                down_outs.append(embeddings_)

                if i == 0:
                    y = torch.sigmoid(self.score_layer(embeddings_, adj_).squeeze())
                else:
                    y = torch.sigmoid(
                        self.score_layer(embeddings_[pre_idx, :], normalize(adj_, add_loop=False)).squeeze())

                score, idx = torch.topk(y, max(2, int(self.ks[i] * adj_.shape[0])))
                _, indices = torch.sort(idx)
                new_score = score[indices]
                new_idx = idx[indices]

                # global node index
                pre_idx = pre_idx[new_idx]
                indices_list.append(pre_idx)

                # for next subgraph selection
                adj_ = adj[pre_idx, :][:, pre_idx]
                mask_score = torch.zeros(n_node).to(features.device)
                mask_score[pre_idx] = new_score
                embeddings_ = torch.mul(embeddings_, torch.unsqueeze(mask_score, -1) + torch.unsqueeze(1 - mask_score,
                                                                                                       -1).detach())  # 这一步有点奇怪，后面一项全是1？
                # -----到这里完成了GP-----

            if self.add_vertical_position:
                vertical_position = torch.zeros(n_node).long().to(adj.device)
                for i in range(self.l_n):
                    vertical_position[indices_list[i]] = int(i + 1)  # 各个节点处于哪一层的位置编码，这个可导吗（估计不行）？
                node_v_pos_embeddings = self.vertival_pos_embedding(vertical_position)
                embeddings_ = torch.cat((embeddings_, node_v_pos_embeddings), dim=-1)  # 这里直接使用embeddings_，没有文中的GNN？
                embeddings_ = F.relu(self.map_v_pos_linear1(embeddings_))
                embeddings_ = F.dropout(embeddings_, self.dropout_v_pos, training=self.training)
                embeddings_ = self.map_v_pos_linear2(embeddings_)

        if self.add_embedding:
            # 默认为true
            embeddings_ += cur_embeddings
        learned_adj = self.metric(embeddings_)
        if self.k:
            # 默认为0
            learned_adj = knn(learned_adj, K=self.k + 1)
        learned_adj = enn(learned_adj, epsilon=self.epsilon)

        if self.l_n > 0:
            for j in reversed(range(self.l_n)):
                learned_adj = symmetry(learned_adj)
                learned_adj = normalize(learned_adj, add_loop=False)
                adj = self.only_modify_subgraph(learned_adj, adj, indices_list[j],
                                                self.stage_fusion_ratio)  # 只修改indices_list[j]部分节点的子结构

                # updata pre_layer subgraph based cur learned subgraph
                embeddings = down_outs[j]
                if self.add_vertical_position:
                    embeddings = torch.cat((embeddings, node_v_pos_embeddings), dim=-1)
                    embeddings = F.relu(self.map_v_pos_linear1(embeddings))
                    embeddings = F.dropout(embeddings, self.dropout_v_pos, training=self.training)
                    embeddings = self.map_v_pos_linear2(embeddings)

                embeddings = self.encoder2(embeddings, normalize(adj, add_loop=False))  # 存在出入，文中是先GNN再融合embeddings

                if self.add_embedding:
                    embeddings += cur_embeddings
                learned_adj = self.metric(embeddings)
                if self.k:
                    learned_adj = knn(learned_adj, K=self.k + 1)
                learned_adj = enn(learned_adj, epsilon=self.epsilon)
        learned_adj = symmetry(learned_adj)
        learned_adj = normalize(learned_adj, add_loop=False)

        # fuse the origin graph and learn graph, and store
        prediction_adj = self.fusion_ratio * learned_adj + (1 - self.fusion_ratio) * adj
        return learned_adj, prediction_adj  # learned_adj 最后一个层次的adj（全图adj和第一次表征再生成的结构），prediction_adj 前者和全图adj的融合

    def stage_recover_adj(self, cur_small_g, pre_big_g, idx):
        n_nums = idx.shape[0]
        x_index = idx.repeat(n_nums)
        y_index = idx.repeat_interleave(n_nums)
        cur_adj_v = cur_small_g.flatten()
        new_pre_adj = pre_big_g.index_put([x_index, y_index], cur_adj_v)
        return new_pre_adj

    def only_modify_subgraph(self, cur_g, pre_g, idx, fusion_ratio):
        cur_small_g = cur_g[idx, :][:, idx]
        pre_small_g = pre_g[idx, :][:, idx]
        new_small_g = cur_small_g * fusion_ratio + pre_small_g * (1 - fusion_ratio)
        new_g = self.stage_recover_adj(new_small_g, pre_g, idx)
        return new_g


class GCN_Classifer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse,
                 batch_norm):
        super(GCN_Classifer, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            # self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            # for _ in range(num_layers - 2):
            #     self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            # self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
            print("Dont support the sparse yet")
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse
        self.batch_norm = batch_norm

        if self.sparse:
            # self.dropout_adj = SparseDropout(dprob=dropout_adj)
            print("Dont support the sparse yet")
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, Adj):

        if self.sparse:
            Adj = copy.deepcopy(Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            if self.batch_norm:
                x = self.bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            print("Dont support sparse yet")
        else:
            # self.gnn_encoder = GCNEncoder(nfeat=in_dim, nhid=hidden_dim, nclass=emb_dim, n_layers=nlayers, dropout=dropout,
            #                         input_layer=False, output_layer=False, spmm_type=0, act='F.relu')
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        if self.sparse:
            print("Dont support sparse yet")
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))

    def forward(self,x, Adj_, branch=None):

        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        # x = self.gnn_encoder(x, Adj)
        z = self.proj_head(x)
        return z, x


class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1


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


