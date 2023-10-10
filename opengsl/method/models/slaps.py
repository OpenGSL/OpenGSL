import torch
import torch.nn.functional as F
# from .GCN3 import GraphConvolution, GCN
import math
import dgl
import numpy as np
from opengsl.method.encoder import GCNEncoder, APPNPEncoder
from opengsl.method.functional import apply_non_linearity, normalize, symmetry, knn
from opengsl.method.metric import InnerProduct
from opengsl.method.transform import KNN


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph

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
            labels = KNN(self.k,metric=self.knn_metric)(self.features)
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
        similarities = InnerProduct()(embeddings)
        similarities = knn(similarities, self.k + 1)
        similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
        return similarities

class GCN_DAE(torch.nn.Module):
    def __init__(self, cfg_model, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k, knn_metric, i_,
                 non_linearity, normalization, mlp_h, mlp_epochs, mlp_act):
        super(GCN_DAE, self).__init__()

        if cfg_model['type'] == 'gcn':
            self.layers = GCNEncoder(in_dim, hidden_dim, nclasses, n_layers=nlayers, dropout=dropout, spmm_type=0)
        elif cfg_model['type'] == 'appnp':
            self.layers = APPNPEncoder(in_dim, hidden_dim, nclasses, spmm_type=1,
                               dropout=dropout, K=cfg_model['appnp_k'], alpha=cfg_model['appnp_alpha'])

        self.dropout_adj = torch.nn.Dropout(p=dropout_adj)
        self.normalization = normalization

        self.graph_gen = MLP(2, features.shape[1], math.floor(math.sqrt(features.shape[1] * mlp_h)),
                                mlp_h, features, mlp_epochs, k, knn_metric, non_linearity, i_,
                                mlp_act).cuda()

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        Adj_ = symmetry(Adj_)
        Adj_ = normalize(Adj_, self.normalization, False)
        return Adj_

    def forward(self, features, x):  # x corresponds to masked_features
        Adj_ = self.get_adj(features)
        Adj = self.dropout_adj(Adj_)

        x = self.layers(x, Adj)

        return x, Adj_

class GCN_C(torch.nn.Module):
    def __init__(self, cfg_model, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GCN_C, self).__init__()

        if cfg_model['type'] == 'gcn':
            self.layers = GCNEncoder(in_channels, hidden_channels, out_channels, n_layers=num_layers, dropout=dropout, spmm_type=0)
        elif cfg_model['type'] == 'appnp':
            self.layers = APPNPEncoder(in_channels, hidden_channels, out_channels, spmm_type=1,
                               dropout=dropout, K=cfg_model['appnp_k'], alpha=cfg_model['appnp_alpha'])

        self.dropout_adj = torch.nn.Dropout(p=dropout_adj)

    def forward(self, x, adj_t):
        Adj = self.dropout_adj(adj_t)

        x = self.layers(x, Adj)
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
