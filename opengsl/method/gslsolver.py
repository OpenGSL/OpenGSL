import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
import random
from copy import deepcopy
from .models.gcn import GCN
from .models.gnn_modules import APPNP, GIN
from .models.grcn import GRCN
from .models.gaug import GAug, eval_edge_pred, MultipleOptimizer, get_lr_schedule_by_sigmoid
from .models.gen import EstimateAdj as GENEstimateAdj, prob_to_adj
from .models.idgl import IDGL, sample_anchors, diff, compute_anchor_adj
from .models.prognn import PGD, prox_operators, EstimateAdj, feature_smoothing
from .models.gt import GT
from .models.slaps import SLAPS
from .models.nodeformer import NodeFormer, adj_mul
from .models.segsl import knn_maxE1, add_knn, get_weight, get_adj_matrix, PartitionTree, get_community, reshape
from .models.sublime import torch_sparse_to_dgl_graph, FGP_learner, ATT_learner, GNN_learner, MLP_learner, GCL, get_feat_mask, split_batch, dgl_graph_to_torch_sparse, GCN_SUB
from .models.stable import DGI, preprocess_adj, aug_random_edge, get_reliable_neighbors
from .models.cogsl import CoGSL
from .models.wsgnn import WSGNN, ELBONCLoss
import torch
import torch.nn.functional as F
import time
from .solver import Solver
from ..utils.utils import get_homophily, sparse_tensor_to_scipy_sparse, scipy_sparse_to_sparse_tensor
from opengsl.data.preprocess.normalize import normalize, normalize_sp_matrix
from ..utils.recorder import Recorder
import dgl
import copy
import os
from os.path import dirname


class GRCNSolver(Solver):
    '''
        A solver to train, evaluate, test GRCN in a run.

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
        >>> conf = opengsl.config.load_conf('grcn', 'cora')
        >>>
        >>> solver = GRCNSolver(conf, dataset)
        >>> # Conduct a experiment run.
        >>> acc, new_structure = solver.run_exp(split=0, debug=True)
        '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "grcn"
        print("Solver Version : [{}]".format("grcn"))
        edge_index = self.adj.coalesce().indices().cpu()
        loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
        edges = torch.cat([edge_index, loop_edge_index], dim=1)
        self.adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [self.n_nodes, self.n_nodes]).to(self.device).coalesce()


    def learn(self, debug=False):
        '''
        Learning process of GRCN.

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
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim1.zero_grad()
            self.optim2.zero_grad()

            # forward and backward
            output, _ = self.model(self.feats, self.adj)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim1.step()
            self.optim2.step()

            # Evaluate
            loss_val, acc_val, adj = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
                if self.conf.analysis['save_graph']:
                    self.best_graph = deepcopy(adj.to_dense())

            # print

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph

    def evaluate(self, test_mask):
        '''
        Evaluation procedure of GRCN.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        adj : torch.tensor        
            The learned structure.
        '''
        self.model.eval()
        with torch.no_grad():
            output, adj = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), adj

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = GRCN(self.n_nodes, self.dim_feats, self.num_targets, self.device, self.conf).to(self.device)
        self.optim1 = torch.optim.Adam(self.model.base_parameters(), lr=self.conf.training['lr'],
                                       weight_decay=self.conf.training['weight_decay'])
        self.optim2 = torch.optim.Adam(self.model.graph_parameters(), lr=self.conf.training['lr_graph'])


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
        self.normalized_adj = normalize(self.adj, sparse=True).to(self.device)
        self.adj_orig = (self.adj.to_dense() + torch.eye(self.n_nodes).to(self.device))  # adj with self loop
        self.conf = conf

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
            hidden, output = self.model.nc_net((self.feats, self.normalized_adj, False))
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            optimizer.step()

            # evaluate
            self.model.eval()
            with torch.no_grad():
                hidden, output = self.model.nc_net((self.feats, self.normalized_adj, False))
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
                hidden, output = self.model.nc_net((self.feats, self.normalized_adj, False))   # the author proposed to validate and test on the original adj
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
                self.best_graph = adj_new.clone().detach()
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

        return self.result, self.best_graph

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


class GENSolver(Solver):
    '''
    A solver to train, evaluate, test GEN in a run.

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
    >>> conf = opengsl.config.load_conf('gen', 'cora')
    >>>
    >>> solver = GENSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "gen"
        print("Solver Version : [{}]".format("gen"))
        self.homophily = get_homophily(self.labels.cpu(), self.adj.to_dense().cpu(), type='node')

    def knn(self, feature):
        # Generate a knn graph for input feature matrix. Note that the graph contains self loop.
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int64)
        dist = cos(feature.detach().cpu().numpy())
        col = np.argpartition(dist, -(self.conf.gsl['k'] + 1), axis=1)[:, -(self.conf.gsl['k'] + 1):].flatten()
        adj[np.arange(self.n_nodes).repeat(self.conf.gsl['k'] + 1), col] = 1
        return adj

    def train_gcn(self, iter, adj, debug=False):
        if debug:
            print('==== Iteration {:04d} ===='.format(iter+1))
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        normalized_adj = normalize(adj, sparse=True)
        for epoch in range(self.conf.training['n_epochs']):
            improve_2 = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            hidden_output, output = self.model((self.feats, normalized_adj, False))
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val, hidden_output, output = self.evaluate(self.val_mask, normalized_adj)

            # save
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                improve_2 = '*'
                if acc_val > self.result['valid']:
                    self.total_time = time.time()-self.start_time
                    improve_1 = '*'
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.best_iter = iter+1
                    self.hidden_output = hidden_output
                    self.output = output if len(output.shape)>1 else output.unsqueeze(1)
                    self.output = F.log_softmax(self.output, dim=1)
                    self.weights = deepcopy(self.model.state_dict())
                    self.best_graph = deepcopy(adj)

            # print
            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve_2))

        print('Iteration {:04d} | Time(s) {:.4f} | Loss(val):{:.4f} | Acc(val):{:.4f} | {}'.format(iter+1,time.time()-t, best_loss_val, best_acc_val, improve_1))

    def structure_learning(self, iter):
        t=time.time()
        self.estimator.reset_obs()
        self.estimator.update_obs(self.knn(self.feats))   # 2
        self.estimator.update_obs(self.knn(self.hidden_output))   # 3
        self.estimator.update_obs(self.knn(self.output))   # 4
        alpha, beta, O, Q, iterations = self.estimator.EM(self.output.max(1)[1].detach().cpu().numpy(), self.conf.gsl['tolerance'])
        adj = torch.tensor(prob_to_adj(Q, self.conf.gsl['threshold']),dtype=torch.float32, device=self.device).to_sparse()
        print('Iteration {:04d} | Time(s) {:.4f} | EM step {:04d}'.format(iter+1,time.time()-t,self.estimator.count))
        return adj

    def learn(self, debug=False):
        '''
        Learning process of GEN.

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
        adj = self.adj

        for iter in range(self.conf.training['n_iters']):
            self.train_gcn(iter, adj, debug)
            adj = self.structure_learning(iter)

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph

    def evaluate(self, test_mask, normalized_adj):
        '''
        Evaluation procedure of GEN.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.
        normalized_adj : torch.tensor
            Adjacency matrix.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        hidden_output : torch.tensor        
            Hidden output of the model.
        output : torch.tensor        
            Output of the model.
        '''
        self.model.eval()
        with torch.no_grad():
            hidden_output, output = self.model((self.feats, normalized_adj, False))
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), hidden_output, output

    def test(self):
        '''
        Test procedure of GEN.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        hidden_output : torch.tensor        
            Hidden output of the model.
        output : torch.tensor        
            Output of the model.
        '''
        self.model.load_state_dict(self.weights)
        normalized_adj = normalize(self.best_graph, sparse=True)
        return self.evaluate(self.test_mask, normalized_adj)

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        if self.conf.model['type']=='gcn':
            self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                             self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                             self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                             self.conf.model['input_layer'], self.conf.model['output_layer'], weight_initializer='glorot',
                             bias_initializer='zeros').to(self.device)
        elif self.conf.model['type']=='appnp':
            self.model = APPNP(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                               dropout=self.conf.model['dropout'], K=self.conf.model['K'],
                               alpha=self.conf.model['alpha']).to(self.device)
        elif self.conf.model['type'] == 'gin':
            self.model = GIN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                               self.conf.model['n_layers'], self.conf.model['mlp_layers']).to(self.device)        
        
        self.estimator = GENEstimateAdj(self.n_classes, self.adj.to_dense(), self.train_mask, self.labels, self.homophily)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.best_iter = 0
        self.hidden_output = None
        self.output = None


class IDGLSolver(Solver):
    '''
    A solver to train, evaluate, test IDGL in a run.

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
    >>> conf = opengsl.config.load_conf('idgl', 'cora')
    >>>
    >>> solver = IDGLSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "idgl"
        print("Solver Version : [{}]".format("idgl"))
        self.conf = conf
        self.run_epoch = self._scalable_run_whole_epoch if self.conf.model['scalable_run'] else self._run_whole_epoch
        if self.conf.model['scalable_run']:
            self.normalized_adj = normalize(self.adj, sparse=True)
        else:
            self.adj = self.adj.to_dense()
            self.normalized_adj = normalize(self.adj, sparse=False)

    def _run_whole_epoch(self, mode='train', debug=False):

        # prepare
        training = mode == 'train'
        if mode == 'train':
            idx = self.train_mask
        elif mode == 'valid':
            idx = self.val_mask
        else:
            idx = self.test_mask
        self.model.train(training)
        network = self.model

        # The first iter
        features = F.dropout(self.feats, self.conf.gsl['feat_adj_dropout'], training=training)
        init_node_vec = features
        init_adj = self.normalized_adj
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, init_node_vec, self.conf.gsl['graph_skip_conn'], graph_include_self=self.conf.gsl['graph_include_self'], init_adj=init_adj)
        # cur_raw_adj是根据输入Z直接产生的adj, cur_adj是前者归一化并和原始adj加权求和的结果
        cur_raw_adj = F.dropout(cur_raw_adj, self.conf.gsl['feat_adj_dropout'], training=training)
        cur_adj = F.dropout(cur_adj, self.conf.gsl['feat_adj_dropout'], training=training)
        if self.conf.model['type'] == 'gcn':
            node_vec, output = network.encoder(init_node_vec, cur_adj)
        else:
            node_vec, output = network.encoder([init_node_vec, cur_adj, False])
        score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
        loss1 = self.loss_fn(output[idx], self.labels[idx])
        loss1 += self.get_graph_loss(cur_raw_adj, init_node_vec)
        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # the following iters
        if training:
            eps_adj = float(self.conf.gsl['eps_adj'])
        else:
            eps_adj = float(self.conf.gsl['test_eps_adj'])
        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj
        loss = 0
        iter_ = 0
        while (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < self.conf.training['max_iter']:
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, self.conf.gsl['graph_skip_conn'], graph_include_self=self.conf.gsl['graph_include_self'], init_adj=init_adj)
            update_adj_ratio = self.conf.gsl['update_adj_ratio']
            cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj   # 这里似乎和论文中有些出入？？
            if self.conf.model['type'] == 'gcn':
                node_vec, output = network.encoder(init_node_vec, cur_adj, self.conf.gsl['gl_dropout'])
            else:
                node_vec, output = network.encoder([init_node_vec, cur_adj, False])
            score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
            loss += self.loss_fn(output[idx], self.labels[idx])
            loss += self.get_graph_loss(cur_raw_adj, init_node_vec)

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, score, cur_adj

    def _scalable_run_whole_epoch(self, mode='train', debug=False):

        # prepare
        training = mode == 'train'
        if mode == 'train':
            idx = self.train_mask
        elif mode == 'valid':
            idx = self.val_mask
        else:
            idx = self.test_mask
        self.model.train(training)
        network = self.model

        # init
        init_adj = self.normalized_adj
        features = F.dropout(self.feats, self.conf.gsl['feat_adj_dropout'], training=training)
        init_node_vec = features
        init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, self.conf.model['num_anchors'])

        # the first iter
        # Compute n x s node-anchor relationship matrix
        cur_node_anchor_adj = network.learn_graph(network.graph_learner, init_node_vec, anchor_features=init_anchor_vec)
        # Compute s x s anchor graph
        cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)
        cur_node_anchor_adj = F.dropout(cur_node_anchor_adj, self.conf.gsl['feat_adj_dropout'], training=training)
        cur_anchor_adj = F.dropout(cur_anchor_adj, self.conf.gsl['feat_adj_dropout'], training=training)

        first_init_agg_vec, init_agg_vec, node_vec, output = network.encoder(init_node_vec, init_adj, cur_node_anchor_adj, self.conf.gsl['graph_skip_conn'])
        anchor_vec = node_vec[sampled_node_idx]
        first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
        score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
        loss1 = self.loss_fn(output[idx], self.labels[idx])
        loss1 += self.get_graph_loss(cur_anchor_adj, init_anchor_vec)

        # the following iters
        if training:
            eps_adj = float(self.conf.gsl['eps_adj'])
        else:
            eps_adj = float(self.conf.gsl['test_eps_adj'])

        pre_node_anchor_adj = cur_node_anchor_adj
        loss = 0
        iter_ = 0
        while (iter_ == 0 or diff(cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj).item() > eps_adj) and iter_ < self.conf.training['max_iter']:
            iter_ += 1
            pre_node_anchor_adj = cur_node_anchor_adj
            # Compute n x s node-anchor relationship matrix
            cur_node_anchor_adj = network.learn_graph(network.graph_learner2, node_vec, anchor_features=anchor_vec)
            # Compute s x s anchor graph
            cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

            update_adj_ratio = self.conf.gsl['update_adj_ratio']
            _, _, node_vec, output = network.encoder(init_node_vec, init_adj, cur_node_anchor_adj, self.conf.gsl['graph_skip_conn'],
                                           first=False, first_init_agg_vec=first_init_agg_vec, init_agg_vec=init_agg_vec, update_adj_ratio=update_adj_ratio,
                                           dropout=self.conf.gsl['gl_dropout'], first_node_anchor_adj=first_node_anchor_adj)
            anchor_vec = node_vec[sampled_node_idx]

            score = self.metric(self.labels[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
            loss += self.loss_fn(output[idx], self.labels[idx])

            loss += self.get_graph_loss(cur_anchor_adj, init_anchor_vec)

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, score, cur_anchor_adj

    def get_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.conf.training['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = torch.ones(out_adj.size(-1)).to(self.device)
        graph_loss += -self.conf.training['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
        graph_loss += self.conf.training['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def learn(self, debug=False):
        '''
        Learning process of IDGL.

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
        wait = 0

        for epoch in range(self.conf.training['max_epochs']):
            t = time.time()
            improve = ''

            # training phase
            loss_train, acc_train, _ = self.run_epoch(mode='train', debug=debug)

            # validation phase
            with torch.no_grad():
                loss_val, acc_val, adj = self.run_epoch(mode='valid', debug=debug)

            if loss_val < self.best_val_loss:
                wait = 0
                self.total_time = time.time()-self.start_time
                self.best_val_loss = loss_val
                self.weights = deepcopy(self.model.state_dict())
                self.result['train'] = acc_train
                self.result['valid'] = acc_val
                improve = '*'
                self.best_graph = deepcopy(adj.clone().detach())
            else:
                wait += 1
                if wait == self.conf.training['patience']:
                    print('Early stop!')
                    break

            # print
            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t, loss_train.item(), acc_train, loss_val, acc_val, improve))

        # test
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        self.model.load_state_dict(self.weights)
        with torch.no_grad():
            loss_test, acc_test, _ = self.run_epoch(mode='test', debug=debug)
        self.result['test']=acc_test
        print(acc_test)
        return self.result, self.best_graph

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = IDGL(self.conf, self.dim_feats, self.num_targets).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])


class PROGNNSolver(Solver):
    '''
    A solver to train, evaluate, test ProGNN in a run.

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
    >>> conf = opengsl.config.load_conf('prognn', 'cora')
    >>>
    >>> solver = PROGNNSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "prognn"
        print("Solver Version : [{}]".format("prognn"))
        self.adj = self.adj.to_dense()

    def train_gcn(self, epoch, debug=False):
        normalized_adj = self.estimator.normalize()

        t = time.time()
        improve = ''
        self.model.train()
        self.optimizer.zero_grad()

        # forward and backward
        output = self.model((self.feats, normalized_adj, False))[-1]
        loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
        acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
        loss_train.backward()
        self.optimizer.step()

        # evaluate
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)
        flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

        # save best model
        if flag:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.best_graph = self.estimator.estimated_adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        if debug:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                epoch+1, time.time() -t, loss_train.item(), acc_train, loss_val, acc_val, improve))

    def train_adj(self, epoch, debug=False):
        estimator = self.estimator
        t = time.time()
        improve = ''
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - self.adj, p='fro')
        normalized_adj = estimator.normalize()

        if self.conf.gsl['lambda_']:
            loss_smooth_feat = feature_smoothing(estimator.estimated_adj, self.feats)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model((self.feats, normalized_adj, False))[-1]
        loss_gcn = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
        acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())

        #loss_symmetric = torch.norm(estimator.estimated_adj - estimator.estimated_adj.t(), p="fro")
        #loss_differential =  loss_fro + self.conf.gamma * loss_gcn + self.conf.lambda_ * loss_smooth_feat + args.phi * loss_symmetric
        loss_differential = loss_fro + self.conf.gsl['gamma'] * loss_gcn + self.conf.gsl['lambda_'] * loss_smooth_feat
        loss_differential.backward()
        self.optimizer_adj.step()
        # we finish the optimization of the differential part above, next we need to do the optimization of loss_l1 and loss_nuclear

        loss_nuclear =  0 * loss_fro
        if self.conf.gsl['beta'] != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                     + self.conf.gsl['gamma'] * loss_gcn \
                     + self.conf.gsl['alpha'] * loss_l1 \
                     + self.conf.gsl['beta'] * loss_nuclear
                     #+ self.conf.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(estimator.estimated_adj.data, min=0, max=1))

        # evaluate
        self.model.eval()
        normalized_adj = estimator.normalize()
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)
        flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

        # save the best model
        if flag:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.best_graph = estimator.estimated_adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        if debug:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss(adj) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                epoch+1, time.time() - t, total_loss.item(), loss_val, acc_val, improve))

    def learn(self, debug=False):
        '''
        Learning process of PROGNN.

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
        for epoch in range(self.conf.training['n_epochs']):
            for i in range(int(self.conf.training['outer_steps'])):
                self.train_adj(epoch, debug=debug)

            for i in range(int(self.conf.training['inner_steps'])):
                self.train_gcn(epoch, debug=debug)

            # we use earlystopping here as prognn is very slow
            if self.improve:
                self.wait = 0
                self.improve = False
            else:
                self.wait += 1
                if self.wait == self.conf.training['patience_iter']:
                    print('Early stop!')
                    break

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph

    def evaluate(self, test_mask, normalized_adj):
        '''
        Evaluation procedure of PROGNN.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.
        normalized_adj : torch.tensor
            Adjacency matrix.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        self.estimator.eval()
        with torch.no_grad():
            logits = self.model((self.feats, normalized_adj, False))[-1]
        logits = logits[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        '''
        Test procedure of PROGNN.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.load_state_dict(self.weights)
        self.estimator.estimated_adj.data.copy_(self.best_graph)
        normalized_adj = self.estimator.normalize()
        return self.evaluate(self.test_mask, normalized_adj)

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        if self.conf.model['type'] == 'gcn':
            self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                             self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                             self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                             self.conf.model['input_layer'], self.conf.model['output_layer'],
                             weight_initializer='uniform').to(self.device)
        elif self.conf.model['type'] == 'appnp':
            self.model = APPNP(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                               dropout=self.conf.model['dropout'], K=self.conf.model['K'],
                               alpha=self.conf.model['alpha']).to(self.device)
        elif self.conf.model['type'] == 'gin':
            self.model = GIN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                               self.conf.model['n_layers'], self.conf.model['mlp_layers']).to(self.device)
        self.estimator = EstimateAdj(self.adj, symmetric=self.conf.gsl['symmetric'], device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                          weight_decay=self.conf.training['weight_decay'])
        self.optimizer_adj = torch.optim.SGD(self.estimator.parameters(), momentum=0.9, lr=self.conf.training['lr_adj'])
        self.optimizer_l1 = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_l1],
                                lr=self.conf.training['lr_adj'], alphas=[self.conf.gsl['alpha']])
        self.optimizer_nuclear = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_nuclear_cuda],
                                     lr=self.conf.training['lr_adj'], alphas=[self.conf.gsl['beta']])

        self.wait = 0


class GTSolver(Solver):
    '''
    A solver to train, evaluate, test GT in a run.

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
    >>> conf = opengsl.config.load_conf('gt', 'cora')
    >>>
    >>> solver = GTSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "gt"
        print("Solver Version : [{}]".format("gt"))
        # prepare dgl graph
        edges = self.adj.coalesce().indices().cpu()
        self.graph = dgl.graph((edges[0], edges[1]), num_nodes=self.n_nodes, idtype=torch.int)
        self.graph = dgl.add_self_loop(self.graph).to(self.device)


    def learn(self, debug=False):
        '''
        Learning process of GRCN.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        0 : constant
        '''
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            x, output, _ = self.model(self.feats, self.graph, self.labels.cpu())

            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val, _ = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                improve = '*'
                self.weights = deepcopy(self.model.state_dict())
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train

            # print

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, homo_heads = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, 0

    def evaluate(self, test_mask, graph_analysis=False):
        '''
        Evaluation procedure of GT.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.
        graph_analysis : bool

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        homo_heads

        '''
        self.model.eval()
        with torch.no_grad():
            x, output, homo_heads = self.model(self.feats, self.graph, self.labels.cpu(), graph_analysis)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), homo_heads

    def test(self):
        '''
        Test procedure of GT.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        homo_heads
        '''
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask, graph_analysis=self.conf.analysis['graph_analysis'])

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = GT(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                   self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm_type'],
                   self.conf.model['n_heads'], self.conf.model['act'], input_layer=self.conf.model['input_layer'],
                        ff=self.conf.model['ff'], output_layer=self.conf.model['output_layer'],
                        use_norm=self.conf.model['use_norm'], use_redisual=self.conf.model['use_residual'],
                        hidden_dim_multiplier=self.conf.model['hidden_dim_multiplier']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                 weight_decay=self.conf.training['weight_decay'])


class SLAPSSolver(Solver):
    '''
        A solver to train, evaluate, test SLAPS in a run.

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
        >>> conf = opengsl.config.load_conf('slaps', 'cora')
        >>>
        >>> solver = SLAPSSolver(conf, dataset)
        >>> # Conduct a experiment run.
        >>> acc, new_structure = solver.run_exp(split=0, debug=True)
        '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "slaps"
        print("Solver Version : [{}]".format("slaps"))

    def learn(self, debug=False):
        '''
        Learning process of SLAPS.

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
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            
            # forward and backward
            output, loss_dae, adj = self.model(self.feats)
            if epoch < self.conf.training['n_epochs'] // self.conf.training['epoch_d']:
                self.model.gcn_c.eval()
                loss_train = self.conf.training['lamda'] * loss_dae
            else:
                loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask]) + self.conf.training['lamda'] * loss_dae 
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()
            
            # Evaluate
            loss_val, acc_val = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
                self.best_graph = adj.clone()

            #print
            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph
    
    def evaluate(self, test_mask):
        '''
        Evaluation procedure of SLAPS.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(self.feats)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = SLAPS(self.n_nodes, self.dim_feats, self.num_targets, self.feats, self.device, self.conf).to(self.device)
        self.optim = torch.optim.Adam([
            {'params': self.model.gcn_c.parameters(), 'lr': self.conf.training['lr'], 'weight_decay': self.conf.training['weight_decay']},
            {'params': self.model.gcn_dae.parameters(), 'lr': self.conf.training['lr_dae'], 'weight_decay': self.conf.training['weight_decay_dae']}
        ])


class NODEFORMERSolver(Solver):
    '''
    A solver to train, evaluate, test Nodeformer in a run.

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
    >>> conf = opengsl.config.load_conf('nodeoformer', 'cora')
    >>>
    >>> solver = NODEFORMERSolverSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "nodeformer"
        print("Solver Version : [{}]".format("nodeformer"))
        edge_index = self.adj.coalesce().indices().cpu()
        loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
        adj = torch.cat([edge_index, loop_edge_index], dim=1).to(self.device)
        self.adjs = []
        self.adjs.append(adj)
        for i in range(conf.model['rb_order'] - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, self.n_nodes)
            self.adjs.append(adj)

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = NodeFormer(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, num_layers=self.conf.model['n_layers'], dropout=self.conf.model['dropout'],
                           num_heads=self.conf.model['n_heads'], use_bn=self.conf.model['use_bn'], nb_random_features=self.conf.model['M'],
                           use_gumbel=self.conf.model['use_gumbel'], use_residual=self.conf.model['use_residual'], use_act=self.conf.model['use_act'],
                           use_jk=self.conf.model['use_jk'],
                           nb_gumbel_sample=self.conf.model['K'], rb_order=self.conf.model['rb_order'], rb_trans=self.conf.model['rb_trans']).to(self.device)
        self.model.reset_parameters()
        self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=self.conf.training['weight_decay'], lr=self.conf.training['lr'])

    def learn(self, debug=False):
        '''
        Learning process of Nodeformer.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        0 : constant
        '''

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output, link_loss = self.model(self.feats, self.adjs, self.conf.model['tau'])
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            loss_train -= self.conf.training['lambda'] * sum(link_loss) / len(link_loss)
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val, = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                improve = '*'
                self.weights = deepcopy(self.model.state_dict())
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train

            # print

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, 0

    def evaluate(self, test_mask):
        '''
        Evaluation procedure of NODEFORMER.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(self.feats, self.adjs, self.conf.model['tau'])
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())


class SEGSLSolver(Solver):
    '''
    A solver to train, evaluate, test SEGSL in a run.

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
    >>> conf = opengsl.config.load_conf('segsl', 'cora')
    >>>
    >>> solver = SEGSLSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "segsl"
        print("Solver Version : [{}]".format("segsl"))

    def learn(self, debug=False):
        '''
        Learning process of SEGSL.

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
        adj = self.adj.to_dense()
        adj.fill_diagonal_(1)
        adj = adj.to_sparse()

        for iter in range(self.conf.training['n_iters']):
            logits = self.train_gcn(iter, adj, debug)
            adj = self.structure_learning(logits, adj)

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph

    def train_gcn(self, iter, adj, debug=False):
        self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                         self.conf.model['dropout'], self.conf.model['input_dropout']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

        if debug:
            print('==== Iteration {:04d} ===='.format(iter+1))
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        normalized_adj = normalize(adj, add_loop=False, sparse=True)
        for epoch in range(self.conf.training['n_epochs']):
            improve_2 = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            hidden_output, output = self.model((self.feats, normalized_adj, False))
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val, hidden_output, output = self.evaluate(self.val_mask, normalized_adj)

            # save
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                improve_2 = '*'
                if acc_val > self.result['valid']:
                    self.total_time = time.time()-self.start_time
                    improve_1 = '*'
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.best_iter = iter+1
                    self.weights = deepcopy(self.model.state_dict())
                    self.best_graph = deepcopy(adj)

            # print
            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve_2))

        print('Iteration {:04d} | Time(s) {:.4f} | Loss(val):{:.4f} | Acc(val):{:.4f} | {}'.format(iter+1,time.time()-t, best_loss_val, best_acc_val, improve_1))

        return output

    def structure_learning(self, logits, adj):
        edge_index = adj.coalesce().indices().t()

        k = knn_maxE1(edge_index, logits)  # edge index对称有自环
        edge_index_2 = add_knn(k, logits, edge_index)
        weight = get_weight(logits, edge_index_2)
        adj_matrix = get_adj_matrix(self.n_nodes, edge_index_2, weight)

        code_tree = PartitionTree(adj_matrix=np.array(adj_matrix))
        code_tree.build_coding_tree(self.conf.gsl['se'])

        community, isleaf = get_community(code_tree)
        new_edge_index = reshape(community, code_tree, isleaf,
                                 self.conf.gsl['k'])
        new_edge_index_2 = reshape(community, code_tree, isleaf,
                                   self.conf.gsl['k'])
        new_edge_index = torch.cat(
            (new_edge_index.t(), new_edge_index_2.t()), dim=0)
        new_edge_index, unique_idx = torch.unique(
            new_edge_index, return_counts=True, dim=0)
        new_edge_index = new_edge_index[unique_idx != 1].t()
        add_num = int(new_edge_index.shape[1])

        new_edge_index = torch.cat(
            (new_edge_index.t(), edge_index.cpu()), dim=0)
        new_edge_index = torch.unique(new_edge_index, dim=0)
        new_edge_index = new_edge_index.t()
        new_weight = get_weight(logits, new_edge_index.t())
        _, delete_idx = torch.topk(new_weight,
                                   k=add_num,
                                   largest=False)
        delete_mask = torch.ones(
            new_edge_index.t().shape[0]).bool()
        delete_mask[delete_idx] = False
        new_edge_index = new_edge_index.t()[delete_mask].t()  # 得到新的edge_index了

        graph = dgl.graph((new_edge_index[0], new_edge_index[1]),
                          num_nodes=self.n_nodes).to(self.device)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        adj = graph.adj().to(self.device)
        return adj

    def evaluate(self, test_mask, normalized_adj):
        '''
        Evaluation procedure of SEGSL.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.
        normalized_adj : torch.tensor
            Adjacency matrix.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        hidden_output : torch.tensor        
            Hidden output of the model.
        output : torch.tensor        
           Output of the model.
        '''
        self.model.eval()
        with torch.no_grad():
            hidden_output, output = self.model((self.feats, normalized_adj, False))
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), hidden_output, output

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.best_iter = 0

    def test(self):
        '''
        Test procedure of SEGSL.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        hidden_output : torch.tensor        
            Hidden output of the model.
        output : torch.tensor        
           Output of the model.
        '''
        self.model.load_state_dict(self.weights)
        normalized_adj = normalize(self.best_graph, add_loop=False, sparse=True)
        return self.evaluate(self.test_mask, normalized_adj)


class SUBLIMESolver(Solver):
    '''
    A solver to train, evaluate, test SUBLIME in a run.

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
    >>> conf = opengsl.config.load_conf('sublime', 'cora')
    >>>
    >>> solver = SUBLIMESolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "sublime"
        print("Solver Version : [{}]".format("sublime"))

    def loss_gcl(self, model, graph_learner, features, anchor_adj):

        # view 1: anchor graph
        if self.conf.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, self.conf.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if self.conf.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, self.conf.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)   # 这个learned adj是有自环的
        if not self.conf.sparse:
            learned_adj = (learned_adj + learned_adj.T) / 2
            learned_adj = normalize(learned_adj, add_loop=False, sparse=False)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if self.conf.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            batches = split_batch(node_idxs, self.conf.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj

    def train_gcn(self, adj, debug=False):
        model = GCN_SUB(nfeat=self.dim_feats, nhid=self.conf.hidden_dim_cls, nclass=self.num_targets,
                        n_layers=self.conf.n_layers_cls, dropout=self.conf.dropout_cls,
                        dropout_adj=self.conf.dropedge_cls, sparse=self.conf.sparse).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.lr_cls, weight_decay=self.conf.w_decay_cls)
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        for epoch in range(self.conf.epochs_cls):
            improve_2 = ''
            t0 = time.time()
            model.train()
            optim.zero_grad()

            # forward and backward
            output = model(self.feats, adj)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(model, self.val_mask, adj)

            # save
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                improve_2 = '*'
                if acc_val > self.result['valid']:
                    self.total_time = time.time()-self.start_time
                    improve_1 = '*'
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.weights = deepcopy(model.state_dict())
                    current_adj = dgl_graph_to_torch_sparse(adj).to_dense() if self.conf.sparse else adj
                    self.best_graph = deepcopy(current_adj)
                    self.best_graph_test = deepcopy(adj)

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve_2))


        print('Time(s) {:.4f} | Loss(val):{:.4f} | Acc(val):{:.4f} | {}'.format(time.time()-t, best_loss_val, best_acc_val, improve_1))

    def evaluate(self, model, test_mask, adj):
        '''
        Evaluation procedure of GRCN.

        Parameters
        ----------
        model : torch.nn.Module
            model.
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.
        adj : torch.tensor
            Adjacency matrix.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        model.eval()
        with torch.no_grad():
            output = model(self.feats, adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        '''
        Test procedure of SUBLIME.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        model = GCN_SUB(nfeat=self.dim_feats, nhid=self.conf.hidden_dim_cls, nclass=self.num_targets,
                        n_layers=self.conf.n_layers_cls, dropout=self.conf.dropout_cls,
                        dropout_adj=self.conf.dropedge_cls, sparse=self.conf.sparse).to(self.device)
        model.load_state_dict(self.weights)
        adj = self.best_graph_test
        return self.evaluate(model, self.test_mask, adj)

    def learn(self, debug=False):
        '''
        Learning process of SUBLIME.

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
        anchor_adj = normalize(self.anchor_adj_raw, add_loop=False, sparse=self.conf.sparse)

        if self.conf.sparse:
            anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
            anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

        for epoch in range(1, self.conf.epochs + 1):

            # Contrastive Learning
            self.model.train()
            self.graph_learner.train()

            loss, Adj = self.loss_gcl(self.model, self.graph_learner, self.feats, anchor_adj)   # Adj是有自环且normalized

            self.optimizer_cl.zero_grad()
            self.optimizer_learner.zero_grad()
            loss.backward()
            self.optimizer_cl.step()
            self.optimizer_learner.step()

            # Structure Bootstrapping
            if (1 - self.conf.tau) and (self.conf.c == 0 or epoch % self.conf.c == 0):
                if self.conf.sparse:
                    learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj).to(self.device)
                    anchor_adj_torch_sparse = anchor_adj_torch_sparse * self.conf.tau \
                                              + learned_adj_torch_sparse * (1 - self.conf.tau)
                    anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                else:
                    anchor_adj = anchor_adj * self.conf.tau + Adj.detach() * (1 - self.conf.tau)

            if debug:
                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

            # Evaluate via Node Classification
            if epoch % self.conf.eval_freq == 0:
                self.model.eval()
                self.graph_learner.eval()
                f_adj = Adj

                if self.conf.sparse:
                    f_adj.edata['w'] = f_adj.edata['w'].detach()
                else:
                    f_adj = f_adj.detach()

                self.train_gcn(f_adj, debug)

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        if self.conf.sparse:
            self.anchor_adj_raw = self.adj
        else:
            self.anchor_adj_raw = self.adj.to_dense()
        anchor_adj = normalize(self.anchor_adj_raw, add_loop=False, sparse=self.conf.sparse)
        if self.conf.type_learner == 'fgp':
            self.graph_learner = FGP_learner(self.feats.cpu(), self.conf.k, self.conf.sim_function, 6, self.conf.sparse)
        elif self.conf.type_learner == 'mlp':
            self.graph_learner = MLP_learner(2, self.feats.shape[1], self.conf.k, self.conf.sim_function, 6, self.conf.sparse,
                                 self.conf.activation_learner)
        elif self.conf.type_learner == 'att':
            self.graph_learner = ATT_learner(2, self.feats.shape[1], self.conf.k, self.conf.sim_function, 6, self.conf.sparse,
                                      self.conf.activation_learner)
        elif self.conf.type_learner == 'gnn':
            self.graph_learner = GNN_learner(2, self.feats.shape[1], self.conf.k, self.conf.sim_function, 6, self.conf.sparse,
                                 self.conf.activation_learner, anchor_adj)
        self.graph_learner = self.graph_learner.to(self.device)
        self.model = GCL(nlayers=self.conf.n_layers, in_dim=self.dim_feats, hidden_dim=self.conf.n_hidden,
                    emb_dim=self.conf.n_embed, proj_dim=self.conf.n_proj,
                    dropout=self.conf.dropout, dropout_adj=self.conf.dropedge_rate, sparse=self.conf.sparse,
                         conf=self.conf).to(self.device)
        self.optimizer_cl = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.wd)
        self.optimizer_learner = torch.optim.Adam(self.graph_learner.parameters(), lr=self.conf.lr,
                                             weight_decay=self.conf.wd)


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
        sp_adj = normalize_sp_matrix(self.processed_adj+(sp.eye(self.n_nodes) * self.conf.beta),
                                  add_loop=False)
        sp_aug_adj1 = normalize_sp_matrix(aug_adj1 + (sp.eye(self.n_nodes) * self.conf.beta),
                                  add_loop=False)
        sp_aug_adj2 = normalize_sp_matrix(aug_adj2 + (sp.eye(self.n_nodes) * self.conf.beta),
                                  add_loop=False)
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
                print(loss)

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
                output = model((feats, adj, True))
            logits = output[test_mask]
            labels = self.labels[test_mask]
            loss = self.loss_fn(logits, labels)
            return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

        def test(model):
            return evaluate(model, self.test_mask)


        model = GCN(self.conf.n_embed, self.conf.n_hidden, self.num_targets, self.conf.n_layers, self.conf.dropout).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        best_loss_val = 10
        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()
            model.train()
            optim.zero_grad()

            # forward and backward
            output = model((feats, adj, True))
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

    def learn(self, debug=False):
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

        normalized_adj_clean = normalize(adj_clean, sparse=False)   # 未使用论文中对归一化的改进
        result = self.train_gcn(embeds, normalized_adj_clean, debug)
        return result, adj_clean

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = DGI(self.dim_feats, self.conf.n_embed, 'prelu').to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.pretrain['lr'], weight_decay=self.conf.pretrain['weight_decay'])
        
        
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
        if self.conf.dataset['init'] :
            _view1 = eval("self."+self.conf.dataset["name_view1"]+"()")
            self.view1_indices = self.get_indices(self.conf.dataset["view1_indices"], _view1, self.conf.dataset["view1_k"])
            _view2 = eval("self."+self.conf.dataset["name_view2"]+"()")
            self.view2_indices = self.get_indices(self.conf.dataset["view2_indices"], _view2, self.conf.dataset["view2_k"])
        else:
            _view1 = sp.load_npz(self.conf.dataset['view1_path'])
            _view2 = sp.load_npz(self.conf.dataset['view2_path'])
            self.view1_indices = torch.load(self.conf.dataset['view1_indices_path'])
            self.view2_indices = torch.load(self.conf.dataset['view2_indices_path'])
        self.view1 = scipy_sparse_to_sparse_tensor(normalize_sp_matrix(_view1, False))
        self.view2 = scipy_sparse_to_sparse_tensor(normalize_sp_matrix(_view2, False))
        self.loss_fn = F.binary_cross_entropy if self.num_targets == 1 else F.nll_loss
        #self.train_mask = np.load('/root/dataset/citeseer/train.npy')
        #self.valid_mask = np.load('/root/dataset/citeseer/val.npy')
        #self.test_mask = np.load('/root/dataset/citeseer/test.npy')
        #print(self.view1_indices.shape)
        #print(self.view1.shape)
        #print(self.view2_indices.shape)
        #print(self.view2.shape)

    def view_knn(self):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int64)
        dist = cos(self.feats.cpu())
        col = np.argpartition(dist, -(self.conf.dataset['knn_k'] + 1), axis=1)[:, -(self.conf.dataset['knn_k'] + 1):].flatten()
        adj[np.arange(self.n_nodes).repeat(self.conf.dataset['knn_k'] + 1), col] = 1
        return sp.coo_matrix(adj)

    def view_adj(self):
        return sparse_tensor_to_scipy_sparse(self.adj)

    def view_diff(self):
        adj = sparse_tensor_to_scipy_sparse(self.adj)
        at = normalize_sp_matrix(adj,False)
        result = self.conf.dataset['diff_alpha'] * sp.linalg.inv(sp.eye(adj.shape[0]) - (1 - self.conf.dataset['diff_alpha']) * at)
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
            view_ = (np.matmul(view_, view.T)>0).astype("int32")
        view_ = torch.tensor(view_).to_sparse()
        #print(view_)
        return view_.indices()

    def topk(self, k, _adj):
        adj = _adj.todense()
        pos = np.zeros(adj.shape)
        for i in range(len(adj)):
            one = adj[i].nonzero()[1]
            if len(one)>k:
                oo = np.argsort(-adj[i, one])
                sele = one[oo[0,:k]]
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
        acc = self.metric(y.cpu().numpy(),output.detach().cpu().numpy())
        return loss, acc

    def train_cls(self):
        new_v1, new_v2 = self.model.get_view(self.view1, self.view1_indices, self.view2, self.view2_indices, self.n_nodes, self.feats)
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
                #with torch.autograd.detect_anomaly():
                loss.backward()
                self.opti_ve.step()
            self.scheduler.step()
            for inner_cls in range(self.conf.training['inner_cls_epoch']):
                self.model.train()
                self.opti_cls.zero_grad()
                cls_loss, _ = self.train_cls()
                #with torch.autograd.detect_anomaly():
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
                self.best_graph = views[0]
            print("EPOCH ",epoch, "\tCUR_LOSS_VAL ", loss_val, "\tCUR_ACC_Val ", acc_val, "\tBEST_ACC_VAL ", self.best_acc_val)
        self.total_time = time.time() - self.start_time
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        #test_f1_macro, test_f1_micro, auc  = self.test()
        self.result['test'] = acc_test
        #print("Test_Macro: ", test_f1_macro, "\tTest_Micro: ", test_f1_micro, "\tAUC: ", auc)
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph.to_dense()


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
                           self.conf.model['mi_hid_1'], self.conf.model['com_lambda_v1'], self.conf.model['com_lambda_v2'],
                           self.conf.model['lam'], self.conf.model['alpha'], self.conf.model['cls_dropout'],
                           self.conf.model['ve_dropout'], self.conf.model['tau'], self.conf.dataset['pyg'],
                           self.conf.dataset['big'], self.conf.dataset['batch'], self.conf.dataset['name']).to(self.device)
        self.opti_ve = torch.optim.Adam(self.model.ve.parameters(), lr=self.conf.training['ve_lr'], weight_decay=self.conf.training['ve_weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opti_ve, 0.99)
        self.opti_cls = torch.optim.Adam(self.model.cls.parameters(), lr=self.conf.training['cls_lr'], weight_decay=self.conf.training['cls_weight_decay'])
        self.opti_mi = torch.optim.Adam(self.model.mi.parameters(), lr=self.conf.training['mi_lr'], weight_decay=self.conf.training['mi_weight_decay'])

        self.view1 = self.view1.to(self.device)
        
        self.view2 = self.view2.to(self.device)
        self.view1_indices = self.view1_indices.to(self.device)
        self.view2_indices = self.view2_indices.to(self.device)

    #def gen_auc_mima(self, logits, label):
    #        preds = torch.argmax(logits, dim=1)
    #        test_f1_macro = f1_score(label.cpu(), preds.cpu(), average='macro')
    #        test_f1_micro = f1_score(label.cpu(), preds.cpu(), average='micro')
    #        
    #        best_proba = F.softmax(logits, dim=1)
    #        if logits.shape[1] != 2:
    #            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
    #                                                    y_score=best_proba.detach().cpu().numpy(),
    #                                                    multi_class='ovr'
    #                                                    )
    #        else:
    #            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
    #                                                    y_score=best_proba[:,1].detach().cpu().numpy()
    #                                                    )
    #        return test_f1_macro, test_f1_micro, auc

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
        self.view = self.best_graph

        return self.evaluate(self.test_mask)



class WSGNNSolver(Solver):
    '''
    A solver to train, evaluate, test WSGNN in a run.

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
    >>> conf = opengsl.config.load_conf('wsgnn', 'cora')
    >>>
    >>> solver = WSGNNSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = 'wsgnn'
        self.edge_index = self.adj.coalesce().indices()


    def learn(self, debug=False):
        '''
        Learning process of WSGNN.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        '''    
        best_node_val = 0
        best_node_test = 0
        best_node_epoch = -1
        for epoch in range(self.conf.training['n_epochs']):
            self.model.train()
            self.optimizer.zero_grad()

            p_y, _, q_y, _ = self.model(self.feats, self.n_nodes, self.edge_index)
            if self.num_targets > 1:
                p_y = torch.nn.functional.log_softmax(p_y, dim=1)
                q_y = torch.nn.functional.log_softmax(q_y, dim=1)
            mask = torch.zeros(self.n_nodes, dtype=bool)
            mask[self.train_mask] = 1
            loss = self.criterion(self.labels, mask, p_y, q_y, )
            loss.backward()
            self.optimizer.step()
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), q_y[self.train_mask].detach().cpu().numpy()) 
            loss_val, acc_val = self.evaluate(self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            if flag:
                self.total_time = time.time() - self.start_time
                best_loss = loss_val
                self.result['train'] = acc_train
                self.result['valid'] = acc_val
                self.weights = deepcopy(self.model.state_dict())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train_acc: {100 * acc_train:.2f}%, '
                    f'Valid_acc: {100 * acc_val:.2f}%, ')
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, None



    def evaluate(self, val_mask):
        '''
        Evaluation procedure of CoGSL.

        Parameters
        ----------
        val_mask : torch.tensor

        Returns
        -------
        loss : float
            Evaluation loss.
        '''
        self.model.eval()
        with torch.no_grad():
            p_y, _, q_y, _ = self.model(self.feats, self.n_nodes, self.edge_index)
        if self.num_targets > 1:
            p_y = torch.nn.functional.log_softmax(p_y, dim=1)
            q_y = torch.nn.functional.log_softmax(q_y, dim=1)
        mask = torch.zeros(self.n_nodes, dtype=bool)
        mask[val_mask] = 1
        loss = self.criterion(self.labels, mask, p_y, q_y)
        acc = self.metric(self.labels[val_mask].cpu().numpy(), q_y[val_mask].detach().cpu().numpy())
        return loss, acc

    def set_method(self):
        self.model = WSGNN(self.conf.model['graph_skip_conn'], self.conf.model['n_hidden'], self.conf.model['dropout'], self.conf.model['n_layers'],
                           self.conf.model['graph_learn_num_pers'], self.conf.model['mlp_layers'], self.conf.model['no_bn'], self.dim_feats,self.n_nodes,self.num_targets).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.criterion = ELBONCLoss(binary=(self.num_targets==1))