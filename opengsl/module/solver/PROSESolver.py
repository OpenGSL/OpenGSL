from copy import deepcopy
import torch
import time
from .solver import Solver
import copy
from opengsl.module.model.prose import Stage_GNN_learner, GCN_Classifer, GCL, get_feat_mask, split_batch
from opengsl.module.functional import normalize
from opengsl.module.encoder import GNNEncoder_OpenGSL


class PROSESolver(Solver):
    '''
    A solver to train, evaluate, test PROSE in a run.

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
    >>> conf = opengsl.config.load_conf('prose', 'cora')
    >>>
    >>> solver = PROSESolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "prose"
        print("Solver Version : [{}]".format("prose"))

    def loss_gcl(self, model, graph_learner, features, anchor_adj, mi_type='learn'):

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

        learned_adj, prediction_adj = graph_learner(features, anchor_adj)

        if mi_type == 'learn':
            z2, _ = model(features_v2, learned_adj, 'learner')
        elif mi_type == 'final':
            z2, _ = model(features_v2, prediction_adj, 'learner')

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
        return loss, learned_adj, prediction_adj

    def learn_nc(self, debug=False):
        adj_original = self.adj.to_dense()
        anchor_adj = normalize(adj_original, add_loop=True)  # 归一化

        for epoch in range(self.conf.epochs):
            t0 = time.time()
            improve = ''

            # train
            self.model.train()
            self.graph_learner.train()
            self.classifier.train()

            # head & tail contrasitive loss
            mi_loss, Adj, prediction_adj = self.loss_gcl(self.model, self.graph_learner, self.feats, anchor_adj,
                                                         self.conf.head_tail_mi_type)
            # cls
            logits = self.classifier(self.feats, prediction_adj)
            loss_train = self.loss_fn(logits[self.train_mask], self.labels[self.train_mask], reduction='mean')
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), logits[self.train_mask].detach().cpu().numpy())
            if self.conf.head_tail_mi:
                final_loss = loss_train + mi_loss * self.conf.mi_ratio
            else:
                final_loss = loss_train

            self.optimizer_cl.zero_grad()
            self.optimizer_learner.zero_grad()
            self.optimizer_classifer.zero_grad()
            final_loss.backward()
            self.optimizer_cl.step()
            self.optimizer_learner.step()
            self.optimizer_classifer.step()

            # validate
            if epoch % self.conf.eval_freq == 0:
                self.classifier.eval()
                with torch.no_grad():
                    logits = self.classifier(self.feats, prediction_adj)
                    loss_val = self.loss_fn(logits[self.val_mask], self.labels[self.val_mask], reduction='mean')
                    acc_val = self.metric(self.labels[self.val_mask].cpu().numpy(), logits[self.val_mask].detach().cpu().numpy())
                flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

                if flag:
                    self.total_time = time.time() - self.start_time
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.best_graph = prediction_adj.detach().clone()
                    self.weights = deepcopy(self.classifier.state_dict())
                    improve = '*'
                elif flag_earlystop:
                    break

                if debug:
                    print(
                        "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                            epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

            # Structure Bootstrapping
            if (1 - self.conf.tau) and (self.conf.c == 0 or epoch % self.conf.c == 0):
                anchor_adj = anchor_adj * self.conf.tau + Adj.detach() * (1 - self.conf.tau)

        # test
        self.classifier.load_state_dict(self.weights)
        with torch.no_grad():
            logits = self.classifier(self.feats, self.best_graph)
            loss_test = self.loss_fn(logits[self.test_mask], self.labels[self.test_mask], reduction='mean')
            acc_test = self.metric(self.labels[self.test_mask].cpu().numpy(), logits[self.test_mask].detach().cpu().numpy())

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))
        return self.result, self.adjs

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.graph_learner = Stage_GNN_learner(2, self.feats.shape[1], self.conf.graph_learner_hidden_dim, self.conf.k,
                                          self.conf.sim_function, self.conf.sparse,
                                          self.conf.activation_learner, self.conf.internal_type, self.conf.stage_ks,
                                          self.conf.share_up_gnn,
                                          self.conf.fusion_ratio, self.conf.stage_fusion_ratio, self.conf.epsilon,
                                          self.conf.add_vertical_position, self.conf.v_pos_dim, self.conf.dropout_v_pos,
                                          self.conf.up_gnn_nlayers, self.conf.dropout_up_gnn, self.conf.add_embedding).to(self.device)

        self.model = GCL(nlayers=self.conf.nlayers, in_dim=self.dim_feats, hidden_dim=self.conf.hidden_dim,
                    emb_dim=self.conf.rep_dim, proj_dim=self.conf.proj_dim,
                    dropout=self.conf.dropout, dropout_adj=self.conf.dropedge_rate, sparse=self.conf.sparse).to(self.device)

        self.classifier = GCN_Classifer(in_channels=self.dim_feats, hidden_channels=self.conf.hidden_dim_cls,
                                   out_channels=self.num_targets,
                                   num_layers=self.conf.nlayers_cls,
                                   dropout=self.conf.dropout_cls, dropout_adj=self.conf.dropedge_cls,
                                   sparse=self.conf.sparse,
                                   batch_norm=self.conf.bn_cls).to(self.device)

        self.optimizer_cl = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr_cl, weight_decay=self.conf.w_decay_cl)
        self.optimizer_learner = torch.optim.Adam(self.graph_learner.parameters(), lr=self.conf.lr_gsl,
                                             weight_decay=self.conf.w_decay_gsl)
        self.optimizer_classifer = torch.optim.Adam(self.classifier.parameters(), lr=self.conf.lr_cls,
                                               weight_decay=self.conf.w_decay_cls)