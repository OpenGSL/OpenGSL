import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.autograd import Variable
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from opengsl.module.encoder import GCNEncoder, MLPEncoder


class VIBGSL(nn.Module):
    def __init__(self, num_node_features, n_hidden, num_classes, n_IB, feature_denoise=False, self_loop=False, eps=0.5, use_edge=False):
        super(VIBGSL, self).__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_dim = n_hidden
        self.IB_size = n_IB
        self.feature_denoise = feature_denoise
        self.self_loop = self_loop
        self.eps = eps
        self.use_edge = use_edge

        self.backbone_gnn = GCNEncoder(self.num_node_features, self.IB_size*2, self.hidden_dim, dropout=0, pyg=True, pool='mean')
        self.mlp = MLPEncoder(self.num_node_features, self.hidden_dim, self.hidden_dim, 2, dropout=0, use_bn=False)
        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(self.num_node_features, init_strategy="constant")
        self.classifier = torch.nn.Sequential(nn.Linear(self.IB_size, self.IB_size), nn.ReLU(), nn.Dropout(p=0.5),
                                              nn.Linear(self.IB_size, self.num_classes))

    def __repr__(self):
        return self.__class__.__name__

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def learn_graph(self, node_features, graph_include_self=False):
        # feature denoise
        device = node_features.device
        if self.feature_denoise:
            feat_mask = torch.sigmoid(self.feat_mask)
            std_tensor = torch.ones_like(node_features, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(node_features, dtype=torch.float) - node_features
            z = torch.normal(mean=mean_tensor, std=std_tensor).to(device)
            x = node_features + z * (1 - feat_mask)
        else:
            x = node_features

        # generate graph
        context_fc = torch.relu(self.mlp(x))
        attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))
        # print(attention)
        attention = torch.clamp(attention, 0.01, 0.99)
        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([0.05]).to(attention.device),
                                                     probs=attention).rsample()
        mask = (weighted_adjacency_matrix > self.eps).detach().float()
        new_adj = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)

        # add I
        if graph_include_self:
            new_adj = new_adj + torch.eye(new_adj.size(0)).to(device)
        return x, new_adj

    def reparametrize_n(self, mu, std):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def forward(self, graphs):
        graphs_list = graphs.to_data_list()
        new_graphs_list = []

        for graph in graphs_list:
            x, edge_index = graph.x, graph.edge_index
            new_feature, new_adj = self.learn_graph(node_features=x, graph_include_self=self.self_loop)
            new_edge_index, new_edge_attr = dense_to_sparse(new_adj)

            new_graph = Data(x=new_feature, edge_index=new_edge_index, edge_attr=new_edge_attr)
            new_graphs_list.append(new_graph)
        loader = DataLoader(new_graphs_list, batch_size=len(new_graphs_list))
        batch_data = next(iter(loader))
        graph_embs = self.backbone_gnn(batch_data, use_edge_attr=self.use_edge)   # edge attr没用
        # graph_embs = self.backbone_gnn(batch_data)

        mu = graph_embs[:, :self.IB_size]
        std = F.softplus(graph_embs[:, self.IB_size:]-self.IB_size, beta=1)
        new_graph_embs = self.reparametrize_n(mu, std)
        logits = self.classifier(new_graph_embs)

        return (mu, std), logits, graphs_list, new_graphs_list