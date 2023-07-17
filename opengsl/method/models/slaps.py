import torch
import torch.nn.functional as F
# from .GCN3 import GraphConvolution, GCN
import math
import dgl
from .gcn import GCN, GraphConvolution
from sklearn.neighbors import kneighbors_graph
from .gnn_modules import APPNP
import numpy as np

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj

def normalize(adj, mode):
    EOS = 1e-10
    if mode == "sym":
        inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
        return inv_degree[:, None] * adj
    else:
        exit("wrong norm mode")



def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2

def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape, device='cuda')
    mask[torch.arange(raw_graph.shape[0], device='cuda').view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1), device='cuda')
    rows = torch.zeros(X.shape[0] * (k + 1), device='cuda')
    cols = torch.zeros(X.shape[0] * (k + 1), device='cuda')
    norm_row = torch.zeros(X.shape[0], device='cuda')
    norm_col = torch.zeros(X.shape[0], device='cuda')
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end, device='cuda').view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values

class MLP(torch.nn.Module):
    def __init__(self, nlayers, isize, hsize, osize, features, mlp_epochs, k, knn_metric, non_linearity, i, mlp_act):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()
        if nlayers == 1:
            self.layers.append(torch.nn.Linear(isize, hsize))
        else:
            self.layers.append(torch.nn.Linear(isize, hsize))
            for _ in range(nlayers - 2):
                self.layers.append(torch.nn.Linear(hsize, hsize))
            self.layers.append(torch.nn.Linear(hsize, osize))

        self.input_dim = isize
        self.output_dim = osize
        self.features = features
        self.mlp_epochs = mlp_epochs
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.i = i
        self.mlp_act = mlp_act
        self.mlp_knn_init()

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def mlp_knn_init(self):
        self.layers.to(self.features.device)
        if self.input_dim == self.output_dim:
            print("MLP full")
            for layer in self.layers:
                layer.weight = torch.nn.Parameter(torch.eye(self.input_dim))
        else:
            optimizer = torch.optim.Adam(self.parameters(), 0.01)
            labels = torch.from_numpy(nearest_neighbors(self.features.cpu(), self.k, self.knn_metric)).cuda()
            for epoch in range(1, self.mlp_epochs):
                self.train()
                logits = self.forward(self.features)
                loss = F.mse_loss(logits, labels, reduction='sum')
                if epoch % 10 == 0:
                    print("MLP loss", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = cal_similarity_graph(embeddings)
        similarities = top_k(similarities, self.k + 1)
        similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
        return similarities

class GCN_DAE(torch.nn.Module):
    def __init__(self, cfg_model, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k, knn_metric, i_,
                 non_linearity, normalization, mlp_h, mlp_epochs, mlp_act):
        super(GCN_DAE, self).__init__()

        if cfg_model['type'] == 'gcn':
            self.layers = GCN(in_dim, hidden_dim, nclasses, n_layers=nlayers, dropout=dropout, spmm_type=1)
        elif cfg_model['type'] == 'appnp':
            self.layers = APPNP(in_dim, hidden_dim, nclasses, spmm_type=1,
                               dropout=dropout, K=cfg_model['appnp_k'], alpha=cfg_model['appnp_alpha'])

        self.dropout_adj = torch.nn.Dropout(p=dropout_adj)
        self.normalization = normalization

        self.graph_gen = MLP(2, features.shape[1], math.floor(math.sqrt(features.shape[1] * mlp_h)),
                                mlp_h, features, mlp_epochs, k, knn_metric, non_linearity, i_,
                                mlp_act).cuda()

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        Adj_ = symmetrize(Adj_)
        Adj_ = normalize(Adj_, self.normalization)
        return Adj_

    def forward(self, features, x):  # x corresponds to masked_features
        Adj_ = self.get_adj(features)
        Adj = self.dropout_adj(Adj_)

        x = self.layers((x, Adj, True))

        return x, Adj_

class GCN_C(torch.nn.Module):
    def __init__(self, cfg_model, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GCN_C, self).__init__()

        if cfg_model['type'] == 'gcn':
            self.layers = GCN(in_channels, hidden_channels, out_channels, n_layers=num_layers, dropout=dropout, spmm_type=1)
        elif cfg_model['type'] == 'appnp':
            self.layers = APPNP(in_channels, hidden_channels, out_channels, spmm_type=1,
                               dropout=dropout, K=cfg_model['appnp_k'], alpha=cfg_model['appnp_alpha'])

        self.dropout_adj = torch.nn.Dropout(p=dropout_adj)

    def forward(self, x, adj_t):
        Adj = self.dropout_adj(adj_t)

        x = self.layers((x, Adj, True))
        return x



class SLAPS(torch.nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, features, device, conf):
        super(SLAPS, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.conf = conf

        self.gcn_dae = GCN_DAE(self.conf.model, nlayers=self.conf.model['nlayers_adj'], in_dim=num_features, hidden_dim=self.conf.model['hidden_adj'], nclasses=num_features,
                             dropout=self.conf.model['dropout1'], dropout_adj=self.conf.model['dropout_adj1'],
                             features=features, k=self.conf.model['k'], knn_metric=self.conf.model['knn_metric'], i_=self.conf.model['i'],
                             non_linearity=self.conf.model['non_linearity'], normalization=self.conf.model['normalization'], mlp_h=self.num_features,
                             mlp_epochs=self.conf.model['mlp_epochs'], mlp_act=self.conf.model['mlp_act'])
        self.gcn_c = GCN_C(self.conf.model, in_channels=num_features, hidden_channels=self.conf.model['hidden'], out_channels=num_classes,
                            num_layers=self.conf.model['nlayers'], dropout=self.conf.model['dropout2'], dropout_adj=self.conf.model['dropout_adj2'])


    def forward(self, features):
        loss_dae, Adj = self.get_loss_masked_features(features)
        logits = self.gcn_c(features, Adj)
        if len(logits.shape) > 1:
            logits = logits.squeeze(1)
        
        return logits, loss_dae, Adj

    def get_loss_masked_features(self, features):
        if self.conf.dataset['feat_type'] == 'binary':
            mask = self.get_random_mask_binary(features, self.conf.training['ratio'], self.conf.training['nr'])
            masked_features = features * (1 - mask)

            logits, Adj = self.gcn_dae(features, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        elif self.conf.dataset['feat_type'] == 'continuous':
            mask = self.get_random_mask_continuous(features, self.conf.training['ratio'])
            # noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
            # masked_features = features + (noise * mask)
            masked_features = features * (1 - mask)

            logits, Adj = self.gcn_dae(features, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        else:
            raise ValueError("Wrong feat_type in dataset_configure.")
    
        return loss, Adj
    
    def get_random_mask_binary(self, features, r, nr):
        nones = torch.sum(features > 0.0).float()
        nzeros = features.shape[0] * features.shape[1] - nones
        pzeros = nones / nzeros / r * nr
        probs = torch.zeros(features.shape, device='cuda')
        probs[features == 0.0] = pzeros
        probs[features > 0.0] = 1 / r
        mask = torch.bernoulli(probs)
        return mask

    def get_random_mask_continuous(self, features, r):
        probs = torch.full(features.shape, 1 / r, device='cuda')
        mask = torch.bernoulli(probs)
        return mask
