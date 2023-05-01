from models.gnn_modules import SGC, LPA, MLP, LINK, LINKX
from models.gcn import GCN
from models.jknet import JKNet
from pipeline.solver import Solver
import torch
from utils.utils import normalize_sp_tensor


class SGCSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats, self.normalized_adj

    def set_method(self):
        self.model = SGC(self.dim_feats, self.num_targets, self.conf.model['n_layers']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                           weight_decay=self.conf.training['weight_decay'])
        normalize = normalize_sp_tensor if self.conf.dataset['normalize'] else lambda x, y: x
        self.normalized_adj = normalize(self.adj, add_loop=self.conf.dataset['add_loop'])


class GCNSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats, self.normalized_adj, True

    def set_method(self):
        self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                    self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                    self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                    self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        normalize = normalize_sp_tensor if self.conf.dataset['normalize'] else lambda x, y: x
        self.normalized_adj = normalize(self.adj, add_loop=self.conf.dataset['add_loop'])


class LPASolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.labels, self.normalized_adj, self.train_mask

    def set_method(self):
        self.model = LPA(self.conf.model['n_layers'], self.conf.model['alpha']).to(self.device)
        normalize = normalize_sp_tensor if self.conf.dataset['normalize'] else lambda x, y: x
        self.normalized_adj = normalize(self.adj, add_loop=self.conf.dataset['add_loop'])

    def learn(self, split=None, debug=False):
        y_pred = self.model(self.input_distributer())
        loss_test = self.loss_fn(y_pred[self.test_mask], self.labels[self.test_mask])
        acc_test = self.metric(self.labels[self.test_mask].cpu().numpy(), y_pred[self.test_mask].detach().cpu().numpy())

        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, 0


class MLPSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats

    def set_method(self):
        self.model = MLP(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])


class LINKSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.adj

    def set_method(self):
        self.model = LINK(self.n_nodes, self.num_targets).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])


class LINKXSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats, self.adj

    def set_method(self):
        self.model = LINKX(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'], self.n_nodes).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])


class JKNetSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats, self.normalized_adj, True

    def set_method(self):
        self.model = JKNet(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                    self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                    self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'], self.conf.model['general'],
                    self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        normalize = normalize_sp_tensor if self.conf.dataset['normalize'] else lambda x, y: x
        self.normalized_adj = normalize(self.adj, add_loop=self.conf.dataset['add_loop'])
        