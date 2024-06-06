import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from opengsl.module.encoder import GCNEncoder
from opengsl.module.model.stable import DGI, preprocess_adj, aug_random_edge, get_reliable_neighbors
import torch
import time
from .solver import Solver
from opengsl.utils.utils import sparse_tensor_to_scipy_sparse, scipy_sparse_to_sparse_tensor
from opengsl.module.functional import normalize, normalize_sp_matrix
import copy


class STABLESolver(Solver):
    '''
    A solver to train, evaluate, test Stable in a run.

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
    >>> conf = opengsl.config.load_conf('stable', 'cora')
    >>>
    >>> solver = STABLESolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "stable"
        print("Solver Version : [{}]".format("stable"))
        self.adj = sparse_tensor_to_scipy_sparse(self.adj)
        self.processed_adj = preprocess_adj(self.feats.cpu().numpy(), self.adj, threshold=self.conf.jt)

    def pretrain(self, debug=False):

        # generate 2 augment views
        adj_delete = self.adj - self.processed_adj
        aug_adj1 = aug_random_edge(self.processed_adj, adj_delete=adj_delete, recover_percent=self.conf.recover_percent)  # random drop edges
        aug_adj2 = aug_random_edge(self.processed_adj, adj_delete=adj_delete, recover_percent=self.conf.recover_percent)  # random drop edges
        sp_adj = normalize_sp_matrix(self.processed_adj+(sp.eye(self.n_nodes) * self.conf.beta), add_loop=False)
        sp_aug_adj1 = normalize_sp_matrix(aug_adj1 + (sp.eye(self.n_nodes) * self.conf.beta), add_loop=False)
        sp_aug_adj2 = normalize_sp_matrix(aug_adj2 + (sp.eye(self.n_nodes) * self.conf.beta), add_loop=False)
        sp_adj = scipy_sparse_to_sparse_tensor(sp_adj).to(self.device)
        sp_aug_adj1 = scipy_sparse_to_sparse_tensor(sp_aug_adj1).to(self.device)
        sp_aug_adj2 = scipy_sparse_to_sparse_tensor(sp_aug_adj2).to(self.device)

        # contrastive learning
        weights = None
        wait = 0
        best = 1e9
        best_t = 0
        b_xent = torch.nn.BCEWithLogitsLoss()
        for epoch in range(self.conf.pretrain['n_epochs']):
            self.model.train()
            self.optim.zero_grad()

            idx = np.random.permutation(self.n_nodes)
            shuf_fts = self.feats.unsqueeze(0)[:, idx, :]

            lbl_1 = torch.ones(1, self.n_nodes)
            lbl_2 = torch.zeros(1, self.n_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            logits = self.model(self.feats.unsqueeze(0), shuf_fts, sp_adj, sp_aug_adj1, sp_aug_adj2)
            loss = b_xent(logits, lbl)
            if debug:
                pass

            if loss < best:
                best = loss
                best_t = epoch
                wait = 0
                weights = copy.deepcopy(self.model.state_dict())
            else:
                wait+=1
            if wait == self.conf.pretrain['patience']:
                print('Early stopping!')
                break

            loss.backward()
            self.optim.step()

        print('Loading {}th epoch'.format(best_t))
        self.model.load_state_dict(weights)

        return self.model.embed(self.feats.unsqueeze(0), sp_adj)

    def train_gcn(self, feats, adj, debug=False):
        def evaluate(model, test_mask):
            model.eval()
            with torch.no_grad():
                output = model(feats, adj)
            logits = output[test_mask]
            labels = self.labels[test_mask]
            loss = self.loss_fn(logits, labels)
            return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

        def test(model):
            return evaluate(model, self.test_mask)


        model = GCNEncoder(self.conf.n_embed, self.conf.n_hidden, self.num_targets, self.conf.n_layers, self.conf.dropout).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        best_loss_val = 10
        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()
            model.train()
            optim.zero_grad()

            # forward and backward
            output = model(feats, adj)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            optim.step()

            # Evaluate
            loss_val, acc_val = evaluate(model, self.val_mask)

            # save
            if acc_val > self.result['valid']:
                improve = '*'
                self.total_time = time.time() - self.start_time
                best_loss_val = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                weights = deepcopy(model.state_dict())

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        model.load_state_dict(weights)
        loss_test, acc_test = test(model)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def learn_nc(self, debug=False):
        '''
        Learning process of STABLE.

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
        embeds = self.pretrain(debug)
        embeds = embeds.squeeze(dim=0)

        # prunue the graph
        adj_clean = preprocess_adj(embeds.cpu().numpy(), self.adj, jaccard=False, threshold=self.conf.cos)
        adj_clean = scipy_sparse_to_sparse_tensor(adj_clean).to(self.device).to_dense()
        # add k neighbors
        get_reliable_neighbors(adj_clean, embeds, k=self.conf.k, degree_threshold=self.conf.threshold)
        # 得到的是0-1 无自环的图

        normalized_adj_clean = normalize(adj_clean)   # 未使用论文中对归一化的改进
        result = self.train_gcn(embeds, normalized_adj_clean, debug)
        return result, adj_clean

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = DGI(self.dim_feats, self.conf.n_embed, 'prelu').to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.pretrain['lr'], weight_decay=self.conf.pretrain['weight_decay'])