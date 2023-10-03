from .gcn import GCN
from .gnn_modules import APPNP, GIN
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro as pyro
from opengsl.method.functional import normalize
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, dim_feats, conf):
        super(VGAE, self).__init__()
        self.gae = conf.gsl['gae']
        # self.gcn_base = GraphConvolution(dim_feats, dim_h, with_bias=False)
        # self.gcn_mean = GraphConvolution(dim_h, dim_z, with_bias=False)
        # self.gcn_logstd = GraphConvolution(dim_h, dim_z, with_bias=False)
        self.conv_graph = GCN(dim_feats, conf.gsl['n_hidden'], conf.gsl['n_embed'], conf.gsl['n_layers'],
                              conf.gsl['dropout'], conf.gsl['input_dropout'], conf.gsl['norm'],
                              conf.gsl['n_linear'], conf.gsl['spmm_type'], conf.gsl['act'],
                              conf.gsl['input_layer'], conf.gsl['output_layer'], bias=False,
                              weight_initializer='glorot')

    def forward(self, feats, adj):
        # GCN encoder
        # hidden = self.gcn_base(feats, adj)
        # self.mean = F.relu(self.gcn_mean(hidden, adj))
        _, mean = self.conv_graph((feats, adj, False))
        mean = F.relu(mean)
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = mean
        else:
            # VGAE
            # self.logstd = F.relu(self.gcn_logstd(hidden, adj))
            # gaussian_noise = torch.randn_like(self.mean)
            # sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
            # Z = sampled_Z
            pass
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits


class GAug(nn.Module):
    def __init__(self,
                 dim_feats,
                 n_classes,
                 conf):
        super(GAug, self).__init__()
        self.temperature = conf.gsl['temperature']
        self.alpha = conf.gsl['alpha']
        # edge prediction network
        self.ep_net = VGAE(dim_feats, conf)
        # node classification network
        # self.nc_net = GCN(dim_feats, dim_h, n_classes, dropout=dropout)
        if conf.model['type']=='gcn':
            self.nc_net = GCN(dim_feats, conf.model['n_hidden'], n_classes, conf.model['n_layers'], conf.model['dropout'],
                              conf.model['input_dropout'], conf.model['norm'], conf.model['n_linear'],
                              conf.model['spmm_type'], conf.model['act'], conf.model['input_layer'],
                              conf.model['output_layer'], weight_initializer='glorot', bias_initializer='zeros')
        elif conf.model['type']=='appnp':
            self.nc_net = APPNP(dim_feats, conf.model['n_hidden'], n_classes,
                               dropout=conf.model['dropout'], K=conf.model['K'],
                               alpha=conf.model['alpha'])
        elif conf.model['type']=='gin':            
            self.nc_net = GIN(dim_feats, conf.model['n_hidden'], n_classes, 
                              conf.model['n_layers'], conf.model['mlp_layers'])
            
    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling

        # print(adj_logits)
        # print(edge_probs)
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def forward(self, feats, adj, adj_orig):
        # print(feats)
        # print(adj)
        adj_logits = self.ep_net(feats, adj)
        if self.alpha == 1:
            adj_new = self.sample_adj(adj_logits)
        else:
            adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.alpha)
        adj_new_normed = normalize(adj_new)
        hidden, output = self.nc_net((feats, adj_new_normed, False))
        return output, adj_logits, adj_new


def eval_edge_pred(adj_pred, val_edges, edge_labels):
    logits = adj_pred[val_edges.T]
    logits = np.nan_to_num(logits)
    roc_auc = roc_auc_score(edge_labels, logits)
    ap_score = average_precision_score(edge_labels, logits)
    return roc_auc, ap_score


class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr


def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
    """ schedule the learning rate with the sigmoid function.
    The learning rate will start with near zero and end with near lr """
    factors = torch.FloatTensor(np.arange(n_epochs))
    factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
    factors = torch.sigmoid(factors)
    # range the factors to [0, 1]
    factors = (factors - factors[0]) / (factors[-1] - factors[0])
    lr_schedule = factors * lr
    return lr_schedule