import torch.nn as nn
import torch
import scipy.sparse as sp
import numpy as np
import copy
import random
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_adj(features, adj, threshold=0.03, jaccard=True):
    """Drop dissimilar edges.(Faster version using numba)
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_triu = sp.triu(adj, format='csr')

    if sp.issparse(features):
        features = features.todense().A  # make it easier for njit processing

    if jaccard:
        removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                       threshold=threshold)
    else:
        removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                      threshold=threshold)
    print('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    return modified_adj


def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)
            if C <= threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def aug_random_edge(input_adj, adj_delete, recover_percent=0.2):
    percent = recover_percent
    adj_delete = sp.tril(adj_delete)
    row_idx, col_idx = adj_delete.nonzero()
    edge_num = int(len(row_idx))
    add_edge_num = int(edge_num * percent)
    print("the number of recovering edges: {:04d}" .format(add_edge_num))
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_list = [(i, j) for i, j in zip(row_idx, col_idx)]
    edge_idx = [i for i in range(edge_num)]
    add_idx = random.sample(edge_idx, add_edge_num)

    for i in add_idx:
        aug_adj[edge_list[i][0]][edge_list[i][1]] = 1
        aug_adj[edge_list[i][1]][edge_list[i][0]] = 1


    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def get_reliable_neighbors(adj, features, k, degree_threshold):
    degree = adj.sum(dim=1)
    degree_mask = degree > degree_threshold
    assert degree_mask.sum().item() >= k
    sim = cosine_similarity(features.to('cpu'))
    sim = torch.FloatTensor(sim).to('cuda')
    sim[:, degree_mask == False] = 0
    _, top_k_indices = sim.topk(k=k, dim=1)
    for i in range(adj.shape[0]):
        adj[i][top_k_indices[i]] = 1
        adj[i][i] = 0
    return


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN_DGI(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    # (features, shuf_fts, aug_features1, aug_features2,
    #  sp_adj if sparse else adj,
    #  sp_aug_adj1 if sparse else aug_adj1,
    #  sp_aug_adj2 if sparse else aug_adj2,
    #  sparse, None, None, None, aug_type=aug_type
    def forward(self, seq1, seq2, adj, aug_adj1, aug_adj2):
        h_0 = self.gcn(seq1, adj)

        h_1 = self.gcn(seq1, aug_adj1)
        h_3 = self.gcn(seq1, aug_adj2)

        c_1 = self.read(h_1)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3)
        c_3 = self.sigm(c_3)

        h_2 = self.gcn(seq2, adj)

        ret1 = self.disc(c_1, h_0, h_2)
        ret2 = self.disc(c_3, h_0, h_2)

        ret = ret1 + ret2   # 这里实际上不符合公式
        return ret

    # Detach the return variables
    def embed(self, seq, adj):
        h_1 = self.gcn(seq, adj)
        # c = self.read(h_1)

        return h_1.detach()


class GCN_DGI(nn.Module):
    # TODO remove in future versions
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN_DGI, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        # 判断一个view和原adj、扰动adj的关系 [1, 2*n_nodes]
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        tmp = self.f_k(h_pl, c_x)
        sc_1 = torch.squeeze(tmp, 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits