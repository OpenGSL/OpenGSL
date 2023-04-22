import torch.nn.functional as F
from copy import deepcopy
from models.idgl import IDGL, sample_anchors, diff, compute_anchor_adj
import torch
import numpy as np
import time
from utils.utils import normalize, normalize_sp_tensor, accuracy
from .solver import BaseSolver

class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("idgl"))
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

    def train(self):
        self.reset()
        self.start_time = time.time()
        wait = 0

        for epoch in range(self.conf.training['max_epochs']):
            t = time.time()
            improve = ''

            # training phase
            loss_train, acc_train, _ = self.run_epoch(mode='train', debug=self.args.debug)

            # validation phase
            with torch.no_grad():
                loss_val, acc_val, adj = self.run_epoch(mode='valid', debug=self.args.debug)

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
            if self.args.debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t, loss_train.item(), acc_train, loss_val, acc_val, improve))

        # test
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        self.model.load_state_dict(self.weights)
        with torch.no_grad():
            loss_test, acc_test, _ = self.run_epoch(mode='test', debug=self.args.debug)
        self.result['test']=acc_test
        print(acc_test)
        return self.result

    def reset(self):
        self.model = IDGL(self.conf, self.dim_feats, self.n_classes)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}

