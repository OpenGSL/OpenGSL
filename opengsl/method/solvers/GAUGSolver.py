import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from opengsl.method.models.gaug import GAug, eval_edge_pred, MultipleOptimizer, get_lr_schedule_by_sigmoid
import torch
import torch.nn.functional as F
import time
from .solver import Solver
from opengsl.method.functional import normalize


class GAUGSolver(Solver):
    '''
    A solver to train, evaluate, test GAug in a run.

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
    >>> conf = opengsl.config.load_conf('gaug', 'cora')
    >>>
    >>> solver = GAUGSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_strcuture = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "gaug"
        print("Solver Version : [{}]".format("gaug"))
        self.normalized_adj = normalize(self.adj)
        self.adj_orig = (self.adj.to_dense() + torch.eye(self.n_nodes).to(self.device))  # adj with self loop

    def pretrain_ep_net(self, norm_w, pos_weight, n_epochs, debug=False):
        """ pretrain the edge prediction network """
        optimizer = torch.optim.Adam(self.model.ep_net.parameters(), lr=self.conf.training['lr'])
        self.model.train()
        for epoch in range(n_epochs):
            t = time.time()
            optimizer.zero_grad()
            adj_logits = self.model.ep_net(self.feats, self.normalized_adj)
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, self.adj_orig, pos_weight=pos_weight)
            if not self.conf.gsl['gae']:
                mu = self.model.ep_net.mean
                lgstd = self.model.ep_net.logstd
                kl_divergence = 0.5/adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                loss -= kl_divergence
            loss.backward()
            optimizer.step()
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            if debug:
                print('EPNet pretrain, Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | auc {:.4f} | ap {:.4f}'
                                 .format(epoch+1, time.time()-t, loss.item(), ep_auc, ep_ap))

    def pretrain_nc_net(self, n_epochs, debug=False):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(self.model.nc_net.parameters(),
                                     lr=self.conf.training['lr'],
                                     weight_decay=self.conf.training['weight_decay'])
        # loss function for node classification
        for epoch in range(n_epochs):
            t = time.time()
            improve = ''
            self.model.train()
            optimizer.zero_grad()

            # forward and backward
            output = self.model.nc_net(self.feats, self.normalized_adj)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            optimizer.step()

            # evaluate
            self.model.eval()
            with torch.no_grad():
                output = self.model.nc_net(self.feats, self.normalized_adj)
                loss_val = self.loss_fn(output[self.val_mask], self.labels[self.val_mask])
            acc_val = self.metric(self.labels[self.val_mask].cpu().numpy(), output[self.val_mask].detach().cpu().numpy())
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                improve = '*'
                self.weights = deepcopy(self.model.state_dict())
                self.best_graph = self.adj.to_dense()

            # print
            if debug:
                print("NCNet pretrain, Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t, loss_train.item(), acc_train, loss_val, acc_val, improve))

    def learn(self, debug=False):
        '''
        Learning process of GAUG.

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
        patience_step = 0

        # prepare
        adj_t = self.adj_orig
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(self.device)

        # pretrain
        self.pretrain_ep_net(norm_w, pos_weight, self.conf.training['pretrain_ep'], debug)
        self.pretrain_nc_net(self.conf.training['pretrain_nc'], debug)

        # train
        optims = MultipleOptimizer(torch.optim.Adam(self.model.ep_net.parameters(),
                                                    lr=self.conf.training['lr']),
                                   torch.optim.Adam(self.model.nc_net.parameters(),
                                                    lr=self.conf.training['lr'],
                                                    weight_decay=self.conf.training['weight_decay']))
        # get the learning rate schedule for the optimizer of ep_net if needed
        if self.conf.training['warmup']:
            ep_lr_schedule = get_lr_schedule_by_sigmoid(self.conf.training['n_epochs'], self.conf.training['lr'], self.conf.training['warmup'])

        for epoch in range(self.conf.training['n_epochs']):
            t = time.time()
            improve = ''
            # update the learning rate for ep_net if needed
            if self.conf.training['warmup']:
                optims.update_lr(0, ep_lr_schedule[epoch])

            self.model.train()
            optims.zero_grad()

            # forward and backward
            output, adj_logits, adj_new = self.model(self.feats, self.normalized_adj, self.adj_orig)
            loss_train = nc_loss = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, self.adj_orig, pos_weight=pos_weight)
            loss_train += self.conf.training['beta'] * ep_loss
            loss_train.backward()
            optims.step()

            # validate
            self.model.eval()
            with torch.no_grad():
                output = self.model.nc_net(self.feats, self.normalized_adj)   # the author proposed to validate and test on the original adj
                loss_val = self.loss_fn(output[self.val_mask], self.labels[self.val_mask])
            acc_val = self.metric(self.labels[self.val_mask].cpu().numpy(), output[self.val_mask].detach().cpu().numpy())
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)

            # save
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
                self.adjs['final'] = adj_new.clone().detach()
                patience_step = 0
            else:
                patience_step += 1
                if patience_step == self.conf.training['patience']:
                    print('Early stop!')
                    break

            # print
            if debug:
                print("Training, Epoch {:05d} | Time(s) {:.4f}".format(epoch+1, time.time() -t))
                print('    EP Loss {:.4f} | EP AUC {:.4f} | EP AP {:.4f}'.format(ep_loss, ep_auc, ep_ap))
                print('    Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}'.format(nc_loss, acc_train, loss_val, acc_val, improve))

        # test
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        self.model.load_state_dict(self.weights)
        with torch.no_grad():
            hidden, output = self.model.nc_net((self.feats, self.normalized_adj, False))
            loss_test = self.loss_fn(output[self.test_mask], self.labels[self.test_mask])
        acc_test = self.metric(self.labels[self.test_mask].cpu().numpy(), output[self.test_mask].detach().cpu().numpy())
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))

        return self.result, self.adjs

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        # sample edges
        if self.labels.size(0) > 5000:
            edge_frac = 0.01
        else:
            edge_frac = 0.1
        adj_matrix = sp.csr_matrix(self.adj.to_dense().cpu().numpy())
        adj_matrix.setdiag(1)  # the original code samples 10%(1%) of the total edges(with self loop)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)
        # sample negative edges
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if i == j:
                continue
            if adj_matrix[i, j] > 0:
                continue
            if (i, j) in added_edges:
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)
        # sample positive edges
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]
        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        self.edge_labels = np.array([1] * n_edges_sample + [0] * n_edges_sample)

        self.model = GAug(self.dim_feats, self.num_targets, self.conf).to(self.device)