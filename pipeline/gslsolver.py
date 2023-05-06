import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
from copy import deepcopy
from models.gcn import GCN
from models.grcn import GRCN
from models.gaug import GAug, eval_edge_pred, MultipleOptimizer
from models.gen import EstimateAdj, prob_to_adj
from models.idgl import IDGL, sample_anchors, diff, compute_anchor_adj
from models.prognn import PGD, prox_operators, EstimateAdj, feature_smoothing
from models.gt import GT
from models.slaps import SLAPS
from models.nodeformer import NodeFormer, adj_mul
from models.segsl import knn_maxE1, add_knn, get_weight, get_adj_matrix, PartitionTree, get_community, reshape
from models.sublime import torch_sparse_to_dgl_graph, FGP_learner, ATT_learner, GNN_learner, MLP_learner, GCL, get_feat_mask, symmetrize, split_batch, dgl_graph_to_torch_sparse, GCN as GCN_sub, accuracy, normalize as normalize_sub
import torch
import torch.nn.functional as F
import time
from pipeline.solver import Solver
from utils.utils import normalize, get_lr_schedule_by_sigmoid, get_homophily, normalize_sp_tensor
import dgl
import copy


class GRCNSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for grcn to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
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
        debug

        Returns
        -------

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
        self.model.eval()
        with torch.no_grad():
            output, adj = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), adj

    def set_method(self):
        self.model = GRCN(self.n_nodes, self.dim_feats, self.num_targets, self.device, self.conf).to(self.device)
        self.optim1 = torch.optim.Adam(self.model.base_parameters(), lr=self.conf.training['lr'],
                                       weight_decay=self.conf.training['weight_decay'])
        self.optim2 = torch.optim.Adam(self.model.graph_parameters(), lr=self.conf.training['lr_graph'])


class GAUGSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for gaug to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        print("Solver Version : [{}]".format("gaug"))
        self.normalized_adj = normalize(self.adj.to_dense()).to(self.device)
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

            # print
            if debug:
                print("NCNet pretrain, Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t, loss_train.item(), acc_train, loss_val, acc_val, improve))

    def learn(self, debug=False):
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
        self.model = GAug(self.dim_feats, self.num_targets, self.conf).to(self.device)

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


class GENSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for gen to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        print("Solver Version : [{}]".format("gen"))
        self.adj = self.adj.to_dense().to(self.device)
        self.homophily = get_homophily(self.labels.cpu().numpy(), self.adj.cpu().numpy())

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
        normalized_adj = normalize(adj)
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
        adj = torch.tensor(prob_to_adj(Q, self.conf.gsl['threshold']),dtype=torch.float32,device=self.device)
        print('Iteration {:04d} | Time(s) {:.4f} | EM step {:04d}'.format(iter+1,time.time()-t,self.estimator.count))
        return adj

    def learn(self, debug=False):

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
        self.model.eval()
        with torch.no_grad():
            hidden_output, output = self.model((self.feats, normalized_adj, False))
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), hidden_output, output

    def test(self):
        self.model.load_state_dict(self.weights)
        normalized_adj = normalize(self.best_graph)
        return self.evaluate(self.test_mask, normalized_adj)

    def set_method(self):
        self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                         self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                         self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                         self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        self.estimator = EstimateAdj(self.num_targets, self.adj, self.train_mask, self.labels, self.homophily)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.best_iter = 0
        self.hidden_output = None
        self.output = None


class IDGLSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for digl to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        print("Solver Version : [{}]".format("idgl"))
        self.conf = conf
        self.normalize = normalize_sp_tensor if self.conf.model['scalable_run'] else normalize
        self.run_epoch = self._scalable_run_whole_epoch if self.conf.model['scalable_run'] else self._run_whole_epoch
        if self.conf.model['scalable_run']:
            self.normalized_adj = self.normalize(self.adj)
        else:
            self.adj = self.adj.to_dense()
            self.normalized_adj = normalize(self.adj)

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
        node_vec, output = network.encoder(init_node_vec, cur_adj)
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
            node_vec, output = network.encoder(init_node_vec, cur_adj, self.conf.gsl['gl_dropout'])
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
                self.best_graph = adj.clone().detach()
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
        return self.result

    def set_method(self):
        self.model = IDGL(self.conf, self.dim_feats, self.num_targets).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])


class PROGNNSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for prognn to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
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

        # save best model
        if loss_val < self.best_val_loss:
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

        # save the best model
        if loss_val < self.best_val_loss:
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

        for epoch in range(self.conf.training['n_epochs']):
            for i in range(int(self.conf.training['outer_steps'])):
                self.train_adj(epoch, debug=debug)

            for i in range(int(self.conf.training['inner_steps'])):
                self.train_gcn(epoch, debug=debug)
            if self.improve:
                self.wait = 0
                self.improve = False
            else:
                self.wait += 1
                if self.wait == self.conf.training['patience']:
                    print('Early stop!')
                    break

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph

    def evaluate(self, test_mask, normalized_adj):
        self.model.eval()
        self.estimator.eval()
        with torch.no_grad():
            logits = self.model((self.feats, normalized_adj, False))[-1]
        logits = logits[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        self.model.load_state_dict(self.weights)
        self.estimator.estimated_adj.data.copy_(self.best_graph)
        normalized_adj = self.estimator.normalize()
        return self.evaluate(self.test_mask, normalized_adj)

    def set_method(self):
        self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                         self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                         self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                         self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
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
    def __init__(self, conf, dataset):
        '''
        Create a solver for gt to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        print("Solver Version : [{}]".format("grcn"))
        # prepare dgl graph
        edges = self.adj.coalesce().indices().cpu()
        self.graph = dgl.graph((edges[0], edges[1]), num_nodes=self.n_nodes, idtype=torch.int)
        self.graph = dgl.add_self_loop(self.graph).to(self.device)


    def learn(self, debug=False):

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            x, output, _ = self.model(self.feats, self.graph, self.labels.cpu().numpy())

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
        self.model.eval()
        with torch.no_grad():
            x, output, homo_heads = self.model(self.feats, self.graph, self.labels.cpu().numpy(), graph_analysis)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), homo_heads

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask, graph_analysis=self.conf.analysis['graph_analysis'])

    def set_method(self):
        self.model = GT(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                   self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm_type'],
                   self.conf.model['n_heads'], self.conf.model['act'], ff=self.conf.model['ff'],
                   hidden_dim_multiplier=self.conf.model['hidden_dim_multiplier']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                 weight_decay=self.conf.training['weight_decay'])


class SLAPSSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for slaps to train, evaluate, test in a run. SLAPS don't use the origin edges from the dataset.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        print("Solver Version : [{}]".format("slaps"))

    def learn(self, debug=False):
        '''
        Learning process of slaps.
        Parameters
        ----------
        debug

        Returns
        -------

        '''

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            
            # forward and backward
            output, loss_dae = self.model(self.feats)
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
        return self.result, 0
    
    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(self.feats)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def set_method(self):
        self.model = SLAPS(self.n_nodes, self.dim_feats, self.num_targets, self.feats, self.device, self.conf).to(self.device)
        self.optim = torch.optim.Adam([
            {'params': self.model.gcn_c.parameters(), 'lr': self.conf.training['lr'], 'weight_decay': self.conf.training['weight_decay']},
            {'params': self.model.gcn_dae.parameters(), 'lr': self.conf.training['lr_dae'], 'weight_decay': self.conf.training['weight_decay_dae']}
        ])


class NODEFORMERSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for grcn to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
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
        self.model = NodeFormer(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, num_layers=self.conf.model['n_layers'], dropout=self.conf.model['dropout'],
                           num_heads=self.conf.model['n_heads'], use_bn=self.conf.model['use_bn'], nb_random_features=self.conf.model['M'],
                           use_gumbel=self.conf.model['use_gumbel'], use_residual=self.conf.model['use_residual'], use_act=self.conf.model['use_act'],
                           use_jk=self.conf.model['use_jk'],
                           nb_gumbel_sample=self.conf.model['K'], rb_order=self.conf.model['rb_order'], rb_trans=self.conf.model['rb_trans']).to(self.device)
        self.model.reset_parameters()
        self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=self.conf.training['weight_decay'], lr=self.conf.training['lr'])

    def learn(self, debug=False):

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
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(self.feats, self.adjs, self.conf.model['tau'])
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())


class SEGSLSolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for segsl to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        print("Solver Version : [{}]".format("segsl"))
        self.normalize = normalize_sp_tensor

    def learn(self, debug=False):

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
                         self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                         self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                         self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

        if debug:
            print('==== Iteration {:04d} ===='.format(iter+1))
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        normalized_adj = self.normalize(adj, add_loop=False)
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
        new_edge_index = torch.concat(
            (new_edge_index.t(), new_edge_index_2.t()), dim=0)
        new_edge_index, unique_idx = torch.unique(
            new_edge_index, return_counts=True, dim=0)
        new_edge_index = new_edge_index[unique_idx != 1].t()
        add_num = int(new_edge_index.shape[1])

        new_edge_index = torch.concat(
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
        self.model.eval()
        with torch.no_grad():
            hidden_output, output = self.model((self.feats, normalized_adj, False))
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), hidden_output, output

    def set_method(self):
        self.best_iter = 0

    def test(self):
        self.model.load_state_dict(self.weights)
        normalized_adj = self.normalize(self.best_graph, add_loop=False)
        return self.evaluate(self.test_mask, normalized_adj)


class SUBLIMESolver(Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for sublime to train, evaluate, test in a run. Some operations are conducted during initialization instead of "set_method" to avoid repetitive computations.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        print("Solver Version : [{}]".format("sublime"))
        self.normalize = normalize_sp_tensor if self.conf.sparse else normalize

    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

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
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize_sub(learned_adj, 'sym', self.conf.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if self.conf.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.conf.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj

    def evaluate_adj_by_cls(self, Adj):

        model = GCN_sub(in_channels=self.dim_feats, hidden_channels=self.conf.hidden_dim_cls, out_channels=self.num_targets, num_layers=self.conf.n_layers_cls,
                    dropout=self.conf.dropout_cls, dropout_adj=self.conf.dropedge_cls, Adj=Adj, sparse=self.conf.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.conf.lr_cls, weight_decay=self.conf.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        model = model.cuda()

        for epoch in range(1, self.conf.epochs_cls + 1):
            model.train()
            loss, accu = self.loss_cls(model, self.train_mask, self.feats, self.labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu = self.loss_cls(model, self.val_mask, self.feats, self.labels)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= self.conf.patience_cls:
                    break
        best_model.eval()
        test_loss, test_accu = self.loss_cls(best_model, self.test_mask, self.feats, self.labels)
        return best_val, test_accu, best_model

    def learn(self, debug=False):
        if self.conf.sparse:
            anchor_adj_raw = self.adj
        else:
            anchor_adj_raw = self.adj.to_dense()

        anchor_adj = normalize_sub(anchor_adj_raw, 'sym', self.conf.sparse)

        if self.conf.sparse:
            anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
            anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

        if self.conf.type_learner == 'fgp':
            graph_learner = FGP_learner(self.feats.cpu(), self.conf.k, self.conf.sim_function, 6, self.conf.sparse)
        elif self.conf.type_learner == 'mlp':
            graph_learner = MLP_learner(2, self.feats.shape[1], self.conf.k, self.conf.sim_function, 6, self.conf.sparse,
                                 self.conf.activation_learner)
        elif self.conf.type_learner == 'att':
            graph_learner = ATT_learner(2, self.feats.shape[1], self.conf.k, self.conf.sim_function, 6, self.conf.sparse,
                                      self.conf.activation_learner)
        elif self.conf.type_learner == 'gnn':
            graph_learner = GNN_learner(2, self.feats.shape[1], self.conf.k, self.conf.sim_function, 6, self.conf.sparse,
                                 self.conf.activation_learner, anchor_adj)

        model = GCL(nlayers=self.conf.n_layers, in_dim=self.dim_feats, hidden_dim=self.conf.n_hidden,
                     emb_dim=self.conf.n_embed, proj_dim=self.conf.n_proj,
                     dropout=self.conf.dropout, dropout_adj=self.conf.dropedge_rate, sparse=self.conf.sparse)

        optimizer_cl = torch.optim.Adam(model.parameters(), lr=self.conf.lr, weight_decay=self.conf.wd)
        optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=self.conf.lr, weight_decay=self.conf.wd)


        model = model.cuda()
        graph_learner = graph_learner.cuda()
        if not self.conf.sparse:
            anchor_adj = anchor_adj.cuda()

        for epoch in range(1, self.conf.epochs + 1):

            model.train()
            graph_learner.train()

            loss, Adj = self.loss_gcl(model, graph_learner, self.feats, anchor_adj)

            optimizer_cl.zero_grad()
            optimizer_learner.zero_grad()
            loss.backward()
            optimizer_cl.step()
            optimizer_learner.step()

            # Structure Bootstrapping
            if (1 - self.conf.tau) and (self.conf.c == 0 or epoch % self.conf.c == 0):
                if self.conf.sparse:
                    learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                    anchor_adj_torch_sparse = anchor_adj_torch_sparse * self.conf.tau \
                                              + learned_adj_torch_sparse * (1 - self.conf.tau)
                    anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                else:
                    anchor_adj = anchor_adj * self.conf.tau + Adj.detach() * (1 - self.conf.tau)

            print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

            if epoch % self.conf.eval_freq == 0:
                model.eval()
                graph_learner.eval()
                f_adj = Adj

                if self.conf.sparse:
                    f_adj.edata['w'] = f_adj.edata['w'].detach()
                else:
                    f_adj = f_adj.detach()

                acc_val, acc_test, _ = self.evaluate_adj_by_cls(f_adj)
                print(acc_val, acc_test)

                if acc_val > self.result['valid']:
                    self.total_time = time.time() - self.start_time
                    improve = '*'
                    self.result['valid'] = acc_val
