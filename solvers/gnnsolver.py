from models.gnn_modules import SGC, LPA, MLP, LINK, LINKX
from models.gcn import GCN
from utils.utils import normalize_sp_tensor
from solvers.solver1 import TaskSolver


class SGCSolver(TaskSolver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats, self.normalized_adj

    def get_model(self):
        model = SGC(self.dim_feats, self.num_targets, self.conf.model['n_layers'])
        return model


class GCNSolver(TaskSolver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats, self.normalized_adj, True

    def get_model(self):
        model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                    self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                    self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                    self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        return model


class LPASolver(TaskSolver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.labels, self.normalized_adj, self.train_mask

    def get_model(self):
        model = LPA(self.conf.model['n_layers'], self.conf.model['alpha'])
        return model

    def learn(self, split=None, debug=False):
        self.prepare(split)
        y_pred = self.model(self.input_distributer())
        loss_test = self.loss_fn(y_pred[self.test_mask], self.labels[self.test_mask])
        acc_test = self.metric(self.labels[self.test_mask].cpu().numpy(), y_pred[self.test_mask].detach().cpu().numpy())

        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def reset(self):
        self.model = self.get_model().to(self.device)
        self.result = {'train': 0, 'valid': 0, 'test': 0}
        self.normalized_adj = self.normalize(self.adj, add_loop=self.conf.dataset['add_loop'])


class MLPSolver(TaskSolver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats

    def get_model(self):
        model = MLP(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'])
        return model


class LINKSolver(TaskSolver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.adj

    def get_model(self):
        model = LINK(self.n_nodes, self.num_targets)
        return model


class LINKXSolver(TaskSolver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)

    def input_distributer(self):
        return self.feats, self.adj

    def get_model(self):
        model = LINKX(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'], self.n_nodes)
        return model







