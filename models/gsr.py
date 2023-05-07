import time
import random
import numpy as np
from gensim.models import Word2Vec
from tqdm import trange
import pickle
import math
import dgl
from tqdm import tqdm
from dgl.nn.pytorch import GraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcn import GCN

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)

def gen_deepwalk_emb(adj, number_walks=10, alpha=0, walk_length=100, window=10, workers=16, size=128):  # ,row, col,
    A = adj.to_dense()
    row, col = A.nonzero(as_tuple=True)
    row, col = row.cpu().numpy(), col.cpu().numpy()
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))

    print("build adj_mat")
    t1 = time.time()
    G = {}
    for [i, j] in edges:
        if i not in G:
            G[i] = []
        if j not in G:
            G[j] = []
        G[i].append(j)
        G[j].append(i)
    for node in G:
        G[node] = list(sorted(set(G[node])))
        if node in G[node]:
            G[node].remove(node)

    nodes = list(sorted(G.keys()))
    print("len(G.keys()):", len(G.keys()), "\tnode_num:", A.shape[0])
    corpus = []  # 存放上下文的 list,每一个节点一个上下文(随机游走序列)
    for cnt in trange(number_walks):
        random.shuffle(nodes)
        for idx, node in enumerate(nodes):
            path = [node]  # 对每个节点找到他的游走序列.
            while len(path) < walk_length:
                cur = path[-1]  # 每次从序列的尾记录当前游走位置.
                if len(G[cur]) > 0:
                    if random.random() >= alpha:
                        path.append(random.choice(G[cur]))  # 如果有邻居,邻接矩阵里随便选一个
                    else:
                        path.append(path[0])  # Random Walk with restart
                else:
                    break
            corpus.append(path)
    t2 = time.time()
    print(f"Corpus generated, time cost: {time2str(t2 - t1)}")
    print("Training word2vec")
    model = Word2Vec(corpus,
                     vector_size=size,  # emb_size
                     window=window,
                     min_count=0,
                     sg=1,  # skip gram
                     hs=1,  # hierarchical softmax
                     workers=workers)
    print("done.., cost: {}s".format(time.time() - t2))
    output = []
    for i in range(A.shape[0]):
        if str(i) in model.wv:  # word2vec 的输出以字典的形式存在.wv 里
            output.append(model.wv[str(i)])
        else:
            print("{} not trained".format(i))
            output.append(np.zeros(size))
    return np.array(output)

class TwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activation, dropout=0.5, is_out_layer=True):
        super().__init__()
        # Fixme: Deal zero degree
        self.conv1 = GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_features, out_features, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(p=dropout)
        self.is_out_layer = is_out_layer
        self.activation = F.elu_ if activation == 'Elu' else F.relu


    def _stochastic_forward(self, blocks, x):
        x = self.activation(self.conv1(blocks[0], x))
        x = self.dropout(x)

        if self.is_out_layer:  # Last layer, no activation and dropout
            x = self.conv2(blocks[1], x)
        else:  # Middle layer, activate and dropout
            x = self.activation(self.conv2(blocks[1], x))
            x = self.dropout(x)
        return x

    def forward(self, g, x, stochastic=False):
        if stochastic:  # Batch forward
            return self._stochastic_forward(g, x)
        else:  # Normal forward
            x = self.activation(self.conv1(g, x))
            x = self.dropout(x)
            if self.is_out_layer:  # Last layer, no activation and dropout
                x = self.conv2(g, x)
            else:  # Middle layer, activate and dropout
                x = self.activation(self.conv2(g, x))
                x = self.dropout(x)
            return x

class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, output_dim, n_hidden, dropout=0.5, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_dim, n_hidden))
        # hidden layers
        for i in range(n_layer - 2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, output_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        return

    def forward(self, input):
        h = input
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            # h = F.relu(layer(h))
            h = self.activation(layer(h))
        return h


def graph_edge_to_lot(g):
    # graph_edge_to list of (row_id, col_id) tuple
    return list(map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist()))

def topk_sim_edges(sim_mat, k, row_start_id, largest):
    v, i = torch.topk(sim_mat.flatten(), k, largest=largest)
    inds = np.array(np.unravel_index(i.cpu().numpy(), sim_mat.shape)).T
    inds[:, 0] = inds[:, 0] + row_start_id
    ind_tensor = torch.tensor(inds).to(sim_mat.device)
    # ret = th.cat((th.tensor(inds).to(sim_mat.device), v.view((-1, 1))), dim=1)
    return ind_tensor, v  # v.view((-1, 1))

def cosine_sim_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    # return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def scalable_graph_refine(g, emb, rm_num, add_num, batch_size, fsim_weight, device, norm=False):
    def _update_topk(sim, start, mask, k, prev_inds, prev_sim, largest):
        # Update TopK similarity and inds
        top_inds, top_sims = topk_sim_edges(sim + mask, k, start, largest)
        temp_inds = torch.cat((prev_inds, top_inds))
        temp_sim = torch.cat((prev_sim, top_sims))
        current_best = temp_sim.topk(k, largest=largest).indices
        return temp_sim[current_best], temp_inds[current_best]

    edges = set(graph_edge_to_lot(g))
    num_batches = int(g.num_nodes() / batch_size) + 1
    if add_num + rm_num == 0:
        return g.edges()

    if norm:
        # Since maximum value of a similarity matrix is fixed as 1, we only have to calculate the minimum value
        fsim_min, ssim_min = 99, 99
        for row_i in tqdm(range(num_batches), desc='Calculating minimum similarity'):
            # ! Initialize batch inds
            start = row_i * batch_size
            end = min((row_i + 1) * batch_size, g.num_nodes())
            if end <= start:
                break

            # ! Calculate similarity matrix
            fsim_min = min(fsim_min, cosine_sim_torch(emb['F'][start:end], emb['F']).min())
            ssim_min = min(ssim_min, cosine_sim_torch(emb['S'][start:end], emb['S']).min())
    # ! Init index and similairty tensor
    # Edge indexes should not be saved as floats in triples, since the number of nodes may well exceeds the maximum of float16 (65504)
    rm_inds, add_inds = [torch.tensor([(0, 0) for i in range(_)]).type(torch.int32).to(device)
                         for _ in [1, 1]]  # Init with one random point (0, 0)
    add_sim = torch.ones(1).type(torch.float16).to(device) * -99
    rm_sim = torch.ones(1).type(torch.float16).to(device) * 99

    for row_i in tqdm(range(num_batches), desc='Batch filtering edges'):
        # ! Initialize batch inds
        start = row_i * batch_size
        end = min((row_i + 1) * batch_size, g.num_nodes())
        if end <= start:
            break

        # ! Calculate similarity matrix
        f_sim = cosine_sim_torch(emb['F'][start:end], emb['F'])
        s_sim = cosine_sim_torch(emb['S'][start:end], emb['S'])
        if norm:
            f_sim = (f_sim - fsim_min) / (1 - fsim_min)
            s_sim = (s_sim - ssim_min) / (1 - ssim_min)
        sim = fsim_weight * f_sim + (1 - fsim_weight) * s_sim

        # ! Get masks
        # Edge mask
        edge_mask, diag_mask = [torch.zeros_like(sim).type(torch.int8) for _ in range(2)]
        row_gids, col_ids = g.out_edges(g.nodes()[start: end])
        row_gids, col_ids = row_gids.long(), col_ids.long()
        edge_mask[row_gids - start, col_ids] = 1
        # Diag mask
        diag_r, diag_c = zip(*[(_ - start, _) for _ in range(start, end)])
        diag_mask[diag_r, diag_c] = 1
        # Add masks: Existing edges and diag edges should be masked
        add_mask = (edge_mask + diag_mask) * -99
        # Remove masks: Non-Existing edges should be masked (diag edges have 1 which is maximum value)
        rm_mask = (1 - edge_mask) * 99

        # ! Update edges to remove and add
        if rm_num > 0:
            k = max(len(rm_sim), rm_num)
            rm_sim, rm_inds = _update_topk(sim, start, rm_mask, k, rm_inds, rm_sim, largest=False)
        if add_num > 0:
            k = max(len(add_sim), add_num)
            add_sim, add_inds = _update_topk(sim, start, add_mask, k, add_inds, add_sim, largest=True)

    # ! Graph refinement
    if rm_num > 0:
        rm_edges = [tuple(_) for _ in rm_inds.cpu().numpy().astype(int).tolist()]
        edges -= set(rm_edges)
    if add_num > 0:
        add_edges = [tuple(_) for _ in add_inds.cpu().numpy().astype(int).tolist()]
        edges |= set(add_edges)
    # assert uf.load_pickle('EdgesGeneratedByOriImplementation') == sorted(edges)
    return edges

class GSR(nn.Module):
    def __init__(self, g, feat_dim, device, conf):
        # ! Initialize variabless
        super(GSR, self).__init__()
        self.g = g
        self.device = device
        self.conf = conf

        # ! Encoder: Pretrained GNN Modules
        self.views = views = ['F', 'S']
        self.encoder = nn.ModuleDict({
            src_view: TwoLayerGCN(feat_dim[src_view], conf.model['n_hidden'], conf.model['n_hidden'],
                                    conf.model['activation'], conf.model['dropout'], is_out_layer=True)
                                        for src_view in views})
        # ! Decoder: Cross-view MLP Mappers
        self.decoder = nn.ModuleDict(
            {f'{src_view}->{tgt_view}':
                 MLP(n_layer=conf.model['decoder_layer'],
                     input_dim=conf.model['n_hidden'],
                     n_hidden=conf.model['decoder_n_hidden'],
                     output_dim=conf.model['n_hidden'], dropout=0,
                     activation=nn.ELU(),
                     )
             for tgt_view in views
             for src_view in views if src_view != tgt_view})

    def forward(self, edge_subgraph, blocks, input, mode='q'):
        def _get_emb(x):
            # Query embedding is stored in source nodes, key embedding in target
            q_nodes, k_nodes = edge_subgraph.edges()
            q_nodes, k_nodes = q_nodes.long(), k_nodes.long()
            return x[q_nodes] if mode == 'q' else x[k_nodes]

        # ! Step 1: Encode node properties to embeddings
        Z = {src_view: _get_emb(encoder(blocks, input[src_view], stochastic=True))
             for src_view, encoder in self.encoder.items()}
        # ! Step 2: Decode embeddings if inter-view
        Z.update({dec: decoder(Z[dec[0]])
                  for dec, decoder in self.decoder.items()})
        return Z

    def refine_graph(self, g, feat):
        '''
        Find the neighborhood candidates for each candidate graph
        :param g: DGL graph
        '''

        # # ! Get Node Property
        emb = {_: self.encoder[_](g.to(self.device), feat[_].to(self.device), stochastic=False).detach()
               for _ in self.views}
        edges = set(graph_edge_to_lot(g))
        rm_num, add_num = [int(float(_) * self.g.num_edges())
                           for _ in (self.conf.refine_graph['rm_ratio'], self.conf.refine_graph['add_ratio'])]

        emb = {k: v.half() for k, v in emb.items()}
        edges = scalable_graph_refine(
            g, emb, rm_num, add_num, self.conf.refine_graph['cos_batch_size'], self.conf.refine_graph['fsim_weight'], self.device, self.conf.refine_graph['fsim_norm'])

        row_id, col_id = map(list, zip(*list(edges)))
        # print(f'High inds {list(high_inds)[:5]}')
        g_new = dgl.add_self_loop(
            dgl.graph((row_id, col_id), num_nodes=self.g.num_nodes())).to(self.device)
        # g_new.ndata['sim'] = sim_adj.to(self.device)
        return g_new

class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, K, T=0.07, device=None):
        super(MemoryMoCo, self).__init__()
        self.device = device
        self.queueSize = K
        self.T = T
        self.index = 0

        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        print("using queue shape: ({},{})".format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()
        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)
        out = torch.cat((l_pos, l_neg), dim=1)

        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).to(self.device)
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out
