import time
from copy import deepcopy
import os
import sys
from torch_geometric.data import Dataset as PygDataset, Data, Batch
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from opengsl.module.encoder import MLPEncoder, GNNEncoder
from opengsl.module.solver import Solver
from opengsl.utils.recorder import Recorder
from opengsl.module.metric import WeightedCosine
from opengsl.module.transform import EpsilonNN, RemoveSelfLoop
from opengsl.module.fuse import Interpolate
# from .gsl import BasicSubGraphLearner, MotifVector, BasicGraphLearner
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import to_dense_adj, subgraph, cumsum, scatter, degree, softmax, coalesce, dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.nn import inits
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn.init as init


# ----------- BFS Parsing -----------
def bfs(adj, s, max_sub_size=10):
    # 从s出发BFS，限制最大节点数为sub_size，得到子图节点列表
    sub = []
    visited = [0 for _ in range(len(adj[0]))]
    queue = [s]
    visited[s] = 1
    node = queue.pop(0)
    sub.append(node)
    while True:
        for x in range(0, len(visited)):
            if adj[node][x] == 1 and visited[x] == 0:
                visited[x] = 1
                queue.append(x)
        if len(queue) == 0:
            break
        else:
            newnode = queue.pop(0)
            node = newnode
            sub.append(node)
            if len(sub) == max_sub_size:
                sub.sort()
                return sub
    sub.sort()
    return sub


def parse_bfs(g, max_n_sub=5, max_sub_size=10, **kwargs):
    # 首先计算度数并根据度数排序
    adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0]
    degrees = degree(g.edge_index[0], g.num_nodes)
    nodes = degrees.sort(descending=True)[1]
    tmp = {}
    subs = []
    for seed in nodes[:max_n_sub]:
        nodes_sub = bfs(adj, int(seed), max_sub_size)
        if tuple(nodes_sub) not in tmp.keys():
            # 防止完全重复的子图
            subs.append(nodes_sub)
            tmp[tuple(nodes_sub)] = 1
    return subs


def original(g, **kwargs):
    device = g.x.device
    n = g.num_nodes
    return [list(range(int(n)))]


# ----------- Prepare Parsed Data -----------
def prepare_parsed_data(dataset, parsing='parse_bfs', parsing_params=None, require_full_edge=False):
    parsing = eval(parsing)
    graphs2subgraphs = {}
    for i in range(len(dataset)):
        graphs2subgraphs[i] = []
    for i, g in tqdm(enumerate(dataset)):
        subs_of_g = []
        subs = parsing(g, **parsing_params)
        for nodes_sub in subs:
            nodes_sub = torch.tensor(nodes_sub, dtype=torch.long)
            x_sub = g.x[nodes_sub]
            edge_sub = subgraph(nodes_sub, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes)[0]
            # mask
            if require_full_edge:
                tmp = torch.arange(nodes_sub.shape[0])
                row, col = torch.meshgrid(tmp, tmp)
                full_edge_index = torch.stack([row.flatten(), col.flatten()])
                g_sub = Data(x=x_sub, edge_index=edge_sub, y=g.y, belong=g.idx, mapping=nodes_sub, full_edge_index=full_edge_index)
            else:
                g_sub = Data(x=x_sub, edge_index=edge_sub, y=g.y, belong=g.idx, mapping=nodes_sub)
            subs_of_g.append(g_sub)
        graphs2subgraphs[i] = subs_of_g
    return graphs2subgraphs


class SimpleParsingModule(nn.Module):
    def __init__(self, parsing='parse_bfs', parsing_params=None, return_new_mapping=True, require_full_edge=False, reparsing=False, **kwargs):
        super(SimpleParsingModule, self).__init__()
        self.parsing = parsing
        self.parsing_params = parsing_params
        self.graphs2subgraphs = None
        self.return_new_mapping = return_new_mapping
        self.require_full_edge = require_full_edge
        self.reparsing = reparsing

    @ torch.no_grad()
    def forward(self, batch, only_idx=False):
        if only_idx:
            device = batch.x.device
            graphs_list = batch.to_data_list()
            batch_subgraphs_nodes_list = []
            batch_belong_list = torch.tensor([], dtype=torch.long).to(device)
            n_nodes_graph = scatter(batch.batch.new_ones(batch.batch.shape[0]), batch.batch)
            cum_n_nodes = cumsum(n_nodes_graph)
            for k, g in enumerate(graphs_list):
                subs_list = self.graphs2subgraphs[g.idx.item()]
                subs_belong = torch.full((len(subs_list),), k, dtype=torch.long).to(device)
                batch_belong_list = torch.cat([batch_belong_list, subs_belong])
                nodes_list = [tmp.mapping.to(device)+cum_n_nodes[k] for tmp in subs_list]
                batch_subgraphs_nodes_list.extend(nodes_list)
            return batch_subgraphs_nodes_list, batch_belong_list

        else:
            device = batch.x.device
            graphs_list = batch.to_data_list()
            batch_subgraphs_list = []
            batch_belong_list = torch.tensor([], dtype=torch.long).to(device)
            for k, g in enumerate(graphs_list):
                subs_list = self.graphs2subgraphs[g.idx.item()]
                subs_belong = torch.full((len(subs_list),), k, dtype=torch.long).to(device)
                batch_subgraphs_list.extend(subs_list)
                batch_belong_list = torch.cat([batch_belong_list, subs_belong])
            batch_subs = Batch.from_data_list(batch_subgraphs_list).to(device)
            if self.return_new_mapping:
                n_nodes_graph = scatter(batch.batch.new_ones(batch.batch.shape[0]), batch.batch)
                cum_n_nodes = cumsum(n_nodes_graph)
                new_mapping = batch_subs.mapping + cum_n_nodes[batch_belong_list[batch_subs.batch]]
            else:
                new_mapping = None
            return batch_subs, batch_belong_list, new_mapping

    @torch.no_grad()
    def init_parsing(self, dataset):
        if not hasattr(dataset, 'attack_type'):
            path = os.path.join(dataset.path, dataset.name, '{}_{}_{}.pickle'.format(dataset.name, self.parsing , '_'.join(list(map(str, list(self.parsing_params.values()))))))
        else:
            path = os.path.join(dataset.path, dataset.name, '{}_{}_{}_{}.pickle'.format(dataset.name, dataset.attack_type, self.parsing, '_'.join(list(map(str, list(self.parsing_params.values()))))))
        if self.reparsing or (not os.path.exists(path)):
            self.graphs2subgraphs = prepare_parsed_data(dataset.data_raw, self.parsing, self.parsing_params, self.require_full_edge)
            with open(path, 'wb') as file:
                pickle.dump(self.graphs2subgraphs, file)
        else:
            with open(path, 'rb') as file:
                self.graphs2subgraphs = pickle.load(file)


class SubgraphSelectModule(nn.Module):
    def __init__(self, n_feat, topk=0.8, min_score=None, conf_encoder=None, conf_estimator=None, select_flag=True, **kwargs):
        super(SubgraphSelectModule, self).__init__()
        assert ((topk is None) ^ (min_score is None))
        self.n_feat = n_feat
        self.topk = topk
        self.select_flag = select_flag
        self.min_score = min_score
        self.conf_encoder = conf_encoder
        self.conf_estimator = conf_estimator
        if self.select_flag:
            if self.conf_encoder['type'] == 'mlp':
                self.encoder = MLPEncoder(n_feat, **conf_encoder)
            elif self.conf_encoder['type'] == 'gnn':
                self.encoder = GNNEncoder(n_feat, **conf_encoder)
            else:
                raise NotImplementedError
            if self.conf_estimator['type'] == 'mlp':
                self.estimator = MLPEncoder(n_feat=conf_encoder['n_class'], n_class=1, **conf_estimator)
            elif self.conf_estimator['type'] == 'projection':
                self.estimator = ProjectionEstimator(n_feat=conf_encoder['n_class'], **conf_estimator)
            else:
                raise NotImplementedError

    def reset_parameters(self):
        if self.select_flag:
            self.encoder.reset_parameters()
            self.estimator.reset_parameters()

    def forward(self, batch_sub, belong, new_mapping):
        if self.select_flag:
            z_sub, z_node = self.encoder(x=batch_sub.x, edge_index=batch_sub.edge_index, batch=batch_sub.batch,
                                         return_before_pool=True)
            score = self.estimator(z_sub, belong).squeeze()   # (n_sub, )
            sub_index = topk(score, self.topk, belong, self.min_score).sort()[0]   # 保证node和subgraph变量顺序一致

            sub_mask = sub_index.new_full((batch_sub.num_graphs,), -1)
            sub_mask[sub_index] = torch.arange(sub_index.shape[0], device=sub_index.device)
            batch_after_select = sub_mask[batch_sub.batch]
            node_index = torch.where(batch_after_select >= 0)[0]
            new_edge_index, _ = filter_adj(batch_sub.edge_index, node_index=node_index, num_nodes=batch_sub.num_nodes, edge_attr=None)
            new_x = batch_sub.x[node_index]
            new_batch = batch_after_select[node_index]
            new_belong = belong[sub_index]
            if new_mapping is not None:
                new_mapping = new_mapping[node_index]   # 这个mapping对应一个batch里的实际位置（加了节点数的累计
            else:
                new_mapping = batch_sub.mapping[node_index]   # 这个mapping对应原来图里的实际位置
            if 'full_edge_index' in batch_sub:
                new_full_edge_index, _ = filter_adj(batch_sub.full_edge_index, node_index=node_index, num_nodes=batch_sub.num_nodes, edge_attr=None)
            else:
                new_full_edge_index = None
            return new_x, z_node[node_index], z_sub[sub_index], new_edge_index, new_batch, new_belong, new_mapping, score[sub_index], new_full_edge_index
        else:
            return batch_sub.x, 0, 0, batch_sub.edge_index, batch_sub.batch, belong, new_mapping, torch.ones(belong.shape).to(belong.device), batch_sub.full_edge_index if 'full_edge_index' in batch_sub else None


class ProjectionEstimator(nn.Module):
    def __init__(self, n_feat, act='sigmoid', **kwargs):
        super(ProjectionEstimator, self).__init__()
        self.n_feat = n_feat
        self.projection_vector = nn.Parameter(torch.rand(n_feat,))
        self.act = activation_resolver(act)
        self.act_name = act
        self.reset_parameters()

    def reset_parameters(self):
        inits.uniform(self.n_feat, self.projection_vector)

    def forward(self, z, batch):
        score = z@self.projection_vector
        if self.act_name == 'softmax':
            # 不确定是否会导致reproducibility问题
            score = softmax(score, batch)
        else:
            score = self.act(score / self.projection_vector.norm(p=2, dim=-1))
        return score


class BasicSubGraphLearner(nn.Module):
    def __init__(self, n_feat, conf_encoder, n_hidden=None, metric_type='weighted_cosine', conf_metric=None,
                 epsilon=0.5, lamb1=0.5, gsl_one_by_one=False, **kwargs):
        super(BasicSubGraphLearner, self).__init__()
        self.n_feat = n_feat
        self.conf_encoder = conf_encoder
        self.encoder_type = conf_encoder['type']
        self.encoder = None
        if self.encoder_type == 'gnn':
            self.encoder = GNNEncoder(n_feat, **conf_encoder)
        elif self.encoder_type == 'mlp':
            self.encoder = MLPEncoder(n_feat, **conf_encoder)
        self.metric_type = metric_type
        self.n_hidden = conf_encoder['n_hidden'] if self.encoder_type else n_hidden
        self.conf_metric = conf_metric
        if self.metric_type == 'weighted_cosine':
            self.metric = WeightedCosine(d_in=self.n_hidden, **conf_metric)
        else:
            raise NotImplementedError
        self.postprocess = [EpsilonNN(epsilon=epsilon), RemoveSelfLoop()]   # 注意如果不one by one，不能使用knn
        self.lamb1 = lamb1
        self.gsl_one_by_one = gsl_one_by_one

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.metric.reset_parameters()

    def forward(self, x, selected_edge_index, selected_batch, selected_mapping, selected_belong, selected_score,
                full_edge_index=None, return_sub=False, raw_edge_index=None, **kwargs):
        raw_edge_attr = x.new_ones((raw_edge_index.shape[1], ))

        if self.encoder_type:
            z_node = self.encoder(x, edge_index=selected_edge_index, batch=selected_batch, return_before_pool=True)
            if isinstance(z_node, tuple):
                z_node = z_node[1]
        else:
            z_node = x
        # normalize score
        score_sum = scatter(selected_score, selected_belong)
        score_sum = score_sum[selected_belong]
        selected_score = selected_score / score_sum

        if self.gsl_one_by_one:
            c = 0
            edge_index_out = selected_edge_index.new_tensor([])
            edge_index_out_subs = selected_edge_index.new_tensor([])
            edge_attr_out = x.new_tensor([])
            edge_attr_out_subs = x.new_tensor([])
            n_nodes_sub = scatter(selected_batch.new_ones(selected_batch.shape[0]), selected_batch)
            for k in range(selected_batch.max()+1):
                z_node_sub = z_node[c:c+n_nodes_sub[k]]
                adj_sub = self.metric(z_node_sub)
                for p in self.postprocess:
                    adj_sub = p(adj_sub)
                edge_index, edge_attr = dense_to_sparse(adj_sub)
                edge_index_out_subs = torch.cat([edge_index_out_subs, edge_index+c], dim=1)   # 注意这里要加上累计节点数
                edge_attr_out_subs = torch.cat([edge_attr_out_subs, edge_attr])
                edge_attr = edge_attr * selected_score[k] * self.lamb1
                edge_index = selected_mapping[edge_index+c]   # 注意这里要加上累计节点数
                c += n_nodes_sub[k]
                edge_index_out = torch.cat([edge_index_out, edge_index], dim=1)
                edge_attr_out = torch.cat([edge_attr_out, edge_attr])
            # fuse with raw adj
            edge_index_out = torch.cat([edge_index_out, raw_edge_index], dim=1)
            edge_attr_out = torch.cat([edge_attr_out, raw_edge_attr * (1 - self.lamb1)])
            edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out)
        else:
            adj_all = self.metric(z_node)
            mask = adj_all.new_zeros(adj_all.size())
            row, col = full_edge_index
            mask[row, col] = 1
            adj_all = adj_all * mask
            for p in self.postprocess:
                adj_all = p(adj_all)
            edge_index_out_subs, edge_attr_out_subs = dense_to_sparse(adj_all)
            weights = selected_score[selected_batch[edge_index_out_subs[0]]]
            edge_attr_out = edge_attr_out_subs * weights * self.lamb1
            edge_index_out = selected_mapping[edge_index_out_subs]
            # fuse with raw adj
            edge_index_out = torch.cat([edge_index_out, raw_edge_index], dim=1)
            edge_attr_out = torch.cat([edge_attr_out, raw_edge_attr * (1-self.lamb1)])
            # coalesce
            # m = n = raw_edge_index.max() + 1
            # edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out, m=m, n=n)
            edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out)
        if return_sub:
            return edge_index_out_subs, edge_attr_out_subs, edge_index_out, edge_attr_out
        else:
            return edge_index_out, edge_attr_out


class BasicGraphLearner(nn.Module):
    def __init__(self, n_feat, conf_encoder, n_hidden=None, metric_type='weighted_cosine', conf_metric=None,
                 epsilon=0.5, lamb1=0.5, gsl_one_by_one=False, **kwargs):
        super(BasicGraphLearner, self).__init__()
        self.n_feat = n_feat
        self.conf_encoder = conf_encoder
        self.encoder_type = conf_encoder['type']
        self.encoder = None
        if self.encoder_type == 'gnn':
            self.encoder = GNNEncoder(n_feat, **conf_encoder)
        elif self.encoder_type == 'mlp':
            self.encoder = MLPEncoder(n_feat, **conf_encoder)
        self.metric_type = metric_type
        self.n_hidden = conf_encoder['n_hidden'] if self.encoder_type else n_hidden
        self.conf_metric = conf_metric
        if self.metric_type == 'weighted_cosine':
            self.metric = WeightedCosine(d_in=self.n_hidden, **conf_metric)
        else:
            raise NotImplementedError
        self.postprocess = [EpsilonNN(epsilon=epsilon), RemoveSelfLoop()]   # 注意如果不one by one，不能使用knn
        self.fuse = Interpolate(lamb1=lamb1)
        self.gsl_one_by_one = gsl_one_by_one

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.metric.reset_parameters()

    def forward(self, x, edge_index, batch, **kwargs):
        # encode
        if self.encoder_type:
            z_node = self.encoder(x, edge_index=edge_index, batch=batch, return_before_pool=True)
            if isinstance(z_node, tuple):
                z_node = z_node[1]
        else:
            z_node = x
        # generage graph
        if self.gsl_one_by_one:
            c = 0
            edge_index_out = edge_index.new_tensor([])
            edge_attr_out = x.new_tensor([])
            n_nodes = scatter(batch.new_ones(batch.shape[0]), batch)
            for k in range(batch.max() + 1):
                z_node_g = z_node[c:c+n_nodes[k]]
                adj_g = self.metric(z_node_g)
                for p in self.postprocess:
                    adj_g = p(adj_g)
                edge_index_g, edge_attr_g = dense_to_sparse(adj_g)
                edge_index_out = torch.cat([edge_index_out, edge_index_g+c], dim=1)   # 注意这里要加上累计节点数
                edge_attr_out = torch.cat([edge_attr_out, edge_attr_g])
                c += n_nodes[k]
        else:
            full_edge_index = self.prepare_full_edge_index(batch)
            adj_all = self.metric(z_node)
            mask = adj_all.new_zeros(adj_all.size())
            row, col = full_edge_index
            mask[row, col] = 1
            adj_all = adj_all * mask
            for p in self.postprocess:
                adj_all = p(adj_all)
            edge_index_out, edge_attr_out = dense_to_sparse(adj_all)
        return edge_index_out, edge_attr_out

    @ torch.no_grad()
    def prepare_full_edge_index(self, batch):
        full_edge_index = batch.new_tensor([])
        n_nodes = scatter(batch.new_ones(batch.shape[0]), batch)
        cum_nodes = cumsum(n_nodes)
        for i in range(len(n_nodes)):
            tmp = torch.arange(cum_nodes[i], cum_nodes[i+1], device=batch.device)
            row, col = torch.meshgrid(tmp, tmp)
            full_edge_index = torch.cat([full_edge_index, torch.stack([row.flatten(), col.flatten()])], dim=1)
        return full_edge_index


class MotifVector(nn.Module):
    def __init__(self, n_hidden, n_motif_per_class, n_class, sim_type='euclidean_1', temperature=0.2,
                 update_type='topk', k2=1, tau=0.99, weighted_mean=False, device=torch.device('cuda'), **kwargs):
        super(MotifVector, self).__init__()
        self.n_hidden = n_hidden
        self.n_motif_per_class = n_motif_per_class
        self.n_class = n_class
        self.n_motif = n_motif_per_class * n_class
        self.Motif_Vector = torch.randn(n_motif_per_class * n_class, n_hidden).to(device)
        self.sim_type = sim_type
        if self.sim_type == 'euclidean_1':
            self.g_sim = self.euclidean_1
        elif self.sim_type == 'euclidean_2':
            self.g_sim = self.euclidean_2
        elif self.sim_type == 'cosine':
            self.g_sim = self.cosine
        else:
            raise NotImplementedError
        self.temperature = temperature
        # mapping
        self.mapping = torch.zeros(self.n_motif, n_class)
        for j in range(self.n_motif):
            self.mapping[j, j // n_motif_per_class] = 1

        self.update_type = update_type
        self.k2 = k2
        self.tau = tau
        self.weighted_mean = weighted_mean

    def reset_parameters(self):
        init.normal_(self.Motif_Vector)

    @ staticmethod
    def euclidean_1(X, M, epsilon=1e-4):
        # ProtGNN
        xp = X @ M.t()
        distance = -2 * xp + torch.sum(X ** 2, dim=1, keepdim=True) + torch.t(torch.sum(M ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance + 1) / (distance + epsilon))
        return similarity, distance

    @staticmethod
    def euclidean_2(X, M):
        # TopExpert
        distance = torch.sum(torch.pow(X.unsqueeze(1) - M, 2), 2)
        similarity = 1.0 / (1.0 + distance)
        return similarity, distance

    @staticmethod
    def euclidean_3(X, M):
        distance = torch.sum(torch.pow(X.unsqueeze(1) - M, 2), 2)
        similarity = -distance
        return similarity, distance

    @ staticmethod
    def cosine(X, M):
        similarity = F.normalize(X, p=2, dim=-1) @ F.normalize(M, p=2, dim=-1).T
        distance = -1 * similarity
        return similarity, distance

    def loss_contrastive(self, z, y):
        # 用于引导子图接近所属类别的某个motif
        device = z.device
        similarity, distance = self.g_sim(z, self.Motif_Vector)
        true_motifs = torch.t(self.mapping.to(device)[:, y].bool())   # (n_sub, n_motif)
        similarities = torch.exp(similarity / self.temperature)
        sim_pos = torch.max(similarities[true_motifs].reshape(-1, self.n_motif_per_class), dim=1)[0]
        sim_neg = similarities[~true_motifs].reshape(-1, (self.n_class - 1) * self.n_motif_per_class)
        loss = -torch.log(sim_pos / (sim_neg.sum(1) + sim_pos)).mean()
        return loss

    @ torch.no_grad()
    def update_motif(self, z, y):
        if self.update_type == 'kmeans':
            device = z.device
            new_M = torch.tensor([]).to(device)
            z = z.detach().cpu().numpy()
            y = y.cpu().numpy()
            for i in range(self.n_class):
                z_i = z[y == i]
                old_M = self.Motif_Vector[self.n_motif_per_class*i:self.n_motif_per_class*(i+1)].detach().cpu().numpy()
                if z_i.shape[0] < self.n_motif_per_class:
                    new_M = torch.cat([new_M, torch.tensor(old_M).to(device)], dim=0)
                else:
                    kmeans = KMeans(n_clusters=self.n_motif_per_class, random_state=0, init=old_M, n_init='auto').fit(z_i)
                    centroids = kmeans.cluster_centers_
                    new_M = torch.cat([new_M, torch.tensor(centroids).to(device)], dim=0)
            new_M = (1-self.tau) * new_M + self.tau * self.Motif_Vector
            self.Motif_Vector.data = new_M.detach().clone()
        else:
            similarity, distance = self.g_sim(z, self.Motif_Vector)
            index_select, sim_select = self.filter_similar_graph(distance, y)
            new_M = torch.tensor([]).to(z.device)
            for j in range(self.n_motif):
                new_motif = z[index_select[j]]
                weights = torch.ones(new_motif.shape[0], device=z.device)
                if self.weighted_mean:
                    weights = weights * F.softmax(sim_select[j], dim=-1)
                else:
                    weights = weights / new_motif.shape[0]
                new_motif = weights.unsqueeze(0) @ new_motif
                updated_motif = self.tau * self.Motif_Vector[j].detach().clone() + (1 - self.tau) * new_motif
                new_M = torch.cat([new_M, updated_motif], dim=0)
            self.Motif_Vector.data = new_M.detach().clone()

    def filter_similar_graph(self, similarity, y):
        device = y.device
        true_motifs = self.mapping.to(device)[:, y].bool().T
        similarity = similarity * true_motifs   # (n_graph, n_motif)
        index_select = []
        sim_select = []
        for j in range(self.n_motif):
            similarity_j = similarity[:, j]
            index = torch.where(similarity_j != 0)[0]
            if self.update_type == 'topk':
                rank = torch.argsort(similarity_j[index], descending=True)
                if self.k2 >= 1:
                    n_select = self.k2
                else:
                    n_select = int(len(rank) * self.k2)
                n_select = min(n_select, len(rank))
                index_select_j = index[rank[:n_select]]
            elif self.update_type == 'thresh':
                index_select_j = index[similarity_j[index] > self.k2]
            elif self.update_type == 'all':
                index_select_j = index
            else:
                raise NotImplementedError
            index_select.append(index_select_j)
            sim_select.append(similarity_j[index_select_j])
        return index_select, sim_select

    @torch.no_grad()
    def init_motif(self, z, y):
        device = z.device
        new_M = torch.tensor([]).to(device)
        z = z.detach().cpu().numpy()
        y = y.cpu().numpy()
        for i in range(self.n_class):
            z_i = z[y == i]
            kmeans = KMeans(n_clusters=self.n_motif_per_class, random_state=0, n_init='auto').fit(z_i)
            centroids = kmeans.cluster_centers_
            new_M = torch.cat([new_M, torch.tensor(centroids).to(device)], dim=0)
        self.Motif_Vector.data = new_M.detach().clone()

    def forward(self):
        pass


class MOSGSL(nn.Module):
    def __init__(self, conf, n_feat, n_class):
        super(MOSGSL, self).__init__()
        self.conf = conf
        self.n_feat = n_feat
        self.n_class = n_class
        self.use_select = self.conf.use_select and self.conf.parsing_module['parsing'] != 'original'
        self.use_gsl = self.conf.use_gsl
        self.use_motif = self.conf.use_motif
        self.use_seperate_encoder = self.conf.use_seperate_encoder and self.use_motif

        # parse
        self.parsing = SimpleParsingModule(**conf.parsing_module, return_new_mapping=conf.use_gsl, require_full_edge=not (conf.gsl_module['gsl_one_by_one']))
        # select
        self.selecting = SubgraphSelectModule(n_feat=self.n_feat, select_flag=self.use_select, **conf.selecting_module)
        # gsl
        self.gsl = BasicSubGraphLearner(n_feat=self.n_feat, n_hidden=self.selecting.conf_encoder['n_hidden'], **conf.gsl_module)
        # classifier
        self.backbone = GNNEncoder(n_feat=self.n_feat, n_class=self.n_class, **conf.backbone)
        self.classifier = MLPEncoder(n_feat=self.selecting.conf_encoder['n_class'], n_class=n_class, **conf.classifier)
        self.backbone2 = GNNEncoder(n_feat=self.n_feat, n_class=self.n_class, **conf.backbone)
        # motif
        device = torch.device('cuda') if not ('use_cpu' in conf and conf.use_cpu) else torch.device('cpu')
        self.motif = MotifVector(n_hidden=self.backbone.output_linear.n_feat, n_class=n_class, device=device, **conf.motif)

        self.forward = self.forward if self.use_gsl else self.forward_wo_gsl

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward_wo_gsl(self, batch, return_sub=False, only_backbone=False):
        if only_backbone:
            return self.backbone(x=batch.x, edge_index=batch.edge_index, batch=batch.batch), 0
        else:
            batch_subs, belong, new_mapping = self.parsing(batch)
            selected_x, selected_z_node, selected_z_sub, selected_edge_index, selected_batch, selected_belong, selected_mapping, selected_score, _ = self.selecting(batch_subs, belong, new_mapping)
            encoder = self.backbone2 if self.use_seperate_encoder else self.backbone
            z_sub = encoder(selected_x, edge_index=selected_edge_index, batch=selected_batch, get_cls=False)
            z_sub_agg = z_sub * selected_score.unsqueeze(1).expand(-1, z_sub.shape[1])
            z_global = global_add_pool(z_sub_agg, batch=selected_belong)
            output = encoder.output_linear(z_global)
            if return_sub:
                return output, 0, z_sub, selected_belong, selected_score
            else:
                return output, 0

    def forward(self, batch, return_sub=False):
        batch_subs, belong, new_mapping = self.parsing(batch)
        selected_x, selected_z_node, selected_z_sub, selected_edge_index, selected_batch, selected_belong, selected_mapping, selected_score, full_edge_index = self.selecting(batch_subs, belong, new_mapping)
        x_gsl = selected_x if self.gsl.encoder_type else selected_z_node

        edge_index_out_subs, edge_attr_out_subs, edge_index_out, edge_attr_out = \
            self.gsl(x_gsl, selected_edge_index=selected_edge_index, selected_batch=selected_batch,
                     selected_mapping=selected_mapping, selected_belong=selected_belong, selected_score=selected_score,
                     full_edge_index=full_edge_index, return_sub=True, y=batch.y[selected_belong],
                     raw_edge_index=batch.edge_index)
        output = self.backbone(batch.x, edge_index=edge_index_out, edge_attr=edge_attr_out, batch=batch.batch)
        loss_con = 0
        if self.use_motif:
            if self.use_seperate_encoder:
                z_sub = self.backbone2(selected_x, edge_index=edge_index_out_subs, edge_attr=edge_attr_out_subs,
                                      batch=selected_batch, get_cls=False)
            else:
                z_sub = self.backbone(selected_x, edge_index=edge_index_out_subs, edge_attr=edge_attr_out_subs,
                                      batch=selected_batch, get_cls=False)
            y_sub = batch.y[selected_belong]
            loss_con = self.motif.loss_contrastive(z_sub, y_sub)
        if return_sub:
            return output, loss_con, z_sub, selected_belong, selected_score
        else:
            return output, loss_con


