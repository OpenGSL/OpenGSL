import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
from copy import deepcopy
from opengsl.module.model.cogsl import CoGSL
import torch
import torch.nn.functional as F
import time
from .solver import Solver
from opengsl.utils.utils import sparse_tensor_to_scipy_sparse, scipy_sparse_to_sparse_tensor
from opengsl.module.functional import normalize_sp_matrix


class COGSLSolver(Solver):
    '''
    A solver to train, evaluate, test CoGSL in a run.

    Parameters
    ----------
    conf : argparse.Namespace
        Config file.
    dataset : opengsl.data.Dataset
        The dataset.

    Attributes
    ----------
    method_name : str
        The name of the method.

    Examples
    --------
    >>> # load dataset
    >>> import opengsl.dataset
    >>> dataset = opengsl.data.Dataset('cora', feat_norm=True)
    >>> # load config file
    >>> import opengsl.config.load_conf
    >>> conf = opengsl.config.load_conf('cogsl', 'cora')
    >>>
    >>> solver = COGSLSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''

    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "cogsl"
        print("Solver Version : [{}]".format("CoGSL"))
        edge_index = self.adj.coalesce().indices().cpu()
        loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
        edges = torch.cat([edge_index, loop_edge_index], dim=1)
        self.adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [self.n_nodes, self.n_nodes]).to(
            self.device).coalesce()
        if self.conf.dataset['init']:
            _view1 = eval("self." + self.conf.dataset["name_view1"] + "()")
            self.view1_indices = self.get_indices(self.conf.dataset["view1_indices"], _view1,
                                                  self.conf.dataset["view1_k"])
            _view2 = eval("self." + self.conf.dataset["name_view2"] + "()")
            self.view2_indices = self.get_indices(self.conf.dataset["view2_indices"], _view2,
                                                  self.conf.dataset["view2_k"])
        else:
            _view1 = sp.load_npz(self.conf.dataset['view1_path'])
            _view2 = sp.load_npz(self.conf.dataset['view2_path'])
            self.view1_indices = torch.load(self.conf.dataset['view1_indices_path'])
            self.view2_indices = torch.load(self.conf.dataset['view2_indices_path'])
        self.view1 = scipy_sparse_to_sparse_tensor(normalize_sp_matrix(_view1, False))
        self.view2 = scipy_sparse_to_sparse_tensor(normalize_sp_matrix(_view2, False))
        self.loss_fn = F.binary_cross_entropy_with_logits if self.num_targets == 1 else F.nll_loss

    def view_knn(self):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int64)
        dist = cos(self.feats.cpu())
        col = np.argpartition(dist, -(self.conf.dataset['knn_k'] + 1), axis=1)[:,
              -(self.conf.dataset['knn_k'] + 1):].flatten()
        adj[np.arange(self.n_nodes).repeat(self.conf.dataset['knn_k'] + 1), col] = 1
        return sp.coo_matrix(adj)

    def view_adj(self):
        return sparse_tensor_to_scipy_sparse(self.adj)

    def view_diff(self):
        adj = sparse_tensor_to_scipy_sparse(self.adj)
        at = normalize_sp_matrix(adj, False)
        result = self.conf.dataset['diff_alpha'] * sp.linalg.inv(
            sp.eye(adj.shape[0]) - (1 - self.conf.dataset['diff_alpha']) * at)
        return result

    def view_sub(self):
        adj = sparse_tensor_to_scipy_sparse(self.adj)
        adj_ = sp.triu(sp.coo_matrix(adj), 1)
        adj_cand = np.array(adj_.nonzero())
        dele_num = int(self.conf.dataset['sub_rate'] * adj_cand.shape[1])
        adj_sele = np.random.choice(np.arange(adj_cand.shape[1]), dele_num, replace=False)
        adj_sele = adj_cand[:, adj_sele]
        adj_new = sp.coo_matrix((np.ones(adj_sele.shape[1]), (adj_sele[0, :], adj_sele[1, :])), shape=adj_.shape)
        adj_new = adj_new + adj_new.T + sp.eye(adj_new.shape[0])
        return adj_new

    def get_khop_indices(self, k, view):
        view = (view.A > 0).astype("int32")
        view_ = view
        for i in range(1, k):
            view_ = (np.matmul(view_, view.T) > 0).astype("int32")
        view_ = torch.tensor(view_).to_sparse()
        # print(view_)
        return view_.indices()

    def topk(self, k, _adj):
        adj = _adj.todense()
        pos = np.zeros(adj.shape)
        for i in range(len(adj)):
            one = adj[i].nonzero()[1]
            if len(one) > k:
                oo = np.argsort(-adj[i, one])
                sele = one[oo[0, :k]]
                pos[i, sele] = adj[i, sele]
            else:
                pos[i, one] = adj[i, one]
        return pos

    def get_indices(self, val, adj, k):
        if (k == 0):
            return self.get_khop_indices(val, sp.coo_matrix((adj)))
        else:
            kn = self.topk(k, adj)
            return self.get_khop_indices(val, sp.coo_matrix((kn)))

    def train_mi(self, x, views):
        vv1, vv2, v1v2 = self.model.get_mi_loss(x, views)
        return self.conf.model['mi_coe'] * v1v2 + (vv1 + vv2) * (1 - self.conf.model['mi_coe']) / 2

    def loss_acc(self, output, y):
        loss = self.loss_fn(output, y)
        acc = self.metric(y.cpu().numpy(), output.detach().cpu().numpy())
        return loss, acc

    def train_cls(self):
        new_v1, new_v2 = self.model.get_view(self.view1, self.view1_indices, self.view2, self.view2_indices,
                                             self.n_nodes, self.feats)
        logits_v1, logits_v2, prob_v1, prob_v2 = self.model.get_cls_loss(new_v1, new_v2, self.feats)
        curr_v = self.model.get_fusion(new_v1, prob_v1, new_v2, prob_v2)
        logits_v = self.model.get_v_cls_loss(curr_v, self.feats)

        views = [curr_v, new_v1, new_v2]

        loss_v1, _ = self.loss_acc(logits_v1[self.train_mask], self.labels[self.train_mask])
        loss_v2, _ = self.loss_acc(logits_v2[self.train_mask], self.labels[self.train_mask])
        loss_v, _ = self.loss_acc(logits_v[self.train_mask], self.labels[self.train_mask])
        return self.conf.model['cls_coe'] * loss_v + (loss_v1 + loss_v2) * (1 - self.conf.model['cls_coe']) / 2, views

    def learn(self, debug=False):
        '''
        Learning process of CoGSL.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        graph : torch.tensor
            The learned structure.
        '''
        self.best_acc_val = 0
        self.best_loss_val = 1e9
        self.best_test = 0
        self.best_v_cls_weight = None
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.conf.training['main_epoch']):
            curr = np.log(1 + self.conf.training['temp_r'] * epoch)
            curr = min(max(0.05, curr), 0.1)
            for inner_ne in range(self.conf.training['inner_ne_epoch']):
                self.model.train()
                self.opti_ve.zero_grad()
                cls_loss, views = self.train_cls()
                mi_loss = self.train_mi(self.feats, views)
                loss = cls_loss - curr * mi_loss
                # with torch.autograd.detect_anomaly():
                loss.backward()
                self.opti_ve.step()
            self.scheduler.step()
            for inner_cls in range(self.conf.training['inner_cls_epoch']):
                self.model.train()
                self.opti_cls.zero_grad()
                cls_loss, _ = self.train_cls()
                # with torch.autograd.detect_anomaly():
                cls_loss.backward()
                self.opti_cls.step()

            for inner_mi in range(self.conf.training['inner_mi_epoch']):
                self.model.train()
                self.opti_mi.zero_grad()
                _, views = self.train_cls()
                mi_loss = self.train_mi(self.feats, views)
                mi_loss.backward()
                self.opti_mi.step()

            self.model.eval()
            _, views = self.train_cls()
            self.view = views[0]

            loss_val, acc_val = self.evaluate(self.val_mask)
            loss_train, acc_train = self.evaluate(self.train_mask)

            if acc_val >= self.best_acc_val and self.best_loss_val > loss_val:
                self.best_acc_val = max(acc_val, self.best_acc_val)
                self.best_loss_val = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.cls.encoder_v.state_dict())
                self.adjs['final'] = views[0].detach().clone()
            if debug:
                print("EPOCH ", epoch, "\tCUR_LOSS_VAL ", loss_val, "\tCUR_ACC_Val ", acc_val, "\tBEST_ACC_VAL ",
                      self.best_acc_val)
        self.total_time = time.time() - self.start_time
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.adjs

    def evaluate(self, test_mask):
        '''
        Evaluation procedure of CoGSL.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.

        Returns
        -------
        loss : float
            Evaluation loss.
        '''
        logits = self.model.get_v_cls_loss(self.view, self.feats)

        return self.loss_acc(logits[test_mask], self.labels[test_mask])

    def set_method(self):
        self.model = CoGSL(self.dim_feats, self.conf.model['cls_hid_1'], self.num_targets, self.conf.model['gen_hid'],
                           self.conf.model['mi_hid_1'], self.conf.model['com_lambda_v1'],
                           self.conf.model['com_lambda_v2'],
                           self.conf.model['lam'], self.conf.model['alpha'], self.conf.model['cls_dropout'],
                           self.conf.model['ve_dropout'], self.conf.model['tau'], self.conf.dataset['pyg'],
                           self.conf.dataset['big'], self.conf.dataset['batch'], self.conf.dataset['name']).to(
            self.device)
        self.opti_ve = torch.optim.Adam(self.model.ve.parameters(), lr=self.conf.training['ve_lr'],
                                        weight_decay=self.conf.training['ve_weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opti_ve, 0.99)
        self.opti_cls = torch.optim.Adam(self.model.cls.parameters(), lr=self.conf.training['cls_lr'],
                                         weight_decay=self.conf.training['cls_weight_decay'])
        self.opti_mi = torch.optim.Adam(self.model.mi.parameters(), lr=self.conf.training['mi_lr'],
                                        weight_decay=self.conf.training['mi_weight_decay'])

        self.view1 = self.view1.to(self.device)

        self.view2 = self.view2.to(self.device)
        self.view1_indices = self.view1_indices.to(self.device)
        self.view2_indices = self.view2_indices.to(self.device)

    def test(self):
        '''
        Test procedure of CoGSL.

        Returns
        -------
        loss : float
            Evaluation loss.
        '''
        self.model.cls.encoder_v.load_state_dict(self.weights)
        self.model.eval()
        self.view = self.adjs['final']

        return self.evaluate(self.test_mask)