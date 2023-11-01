from copy import deepcopy
from opengsl.module.model.wsgnn import WSGNN, ELBONCLoss
import torch
import torch.nn.functional as F
import time
from .solver import Solver


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
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optimizer.zero_grad()

            p_y, adj_p, q_y, adj_q = self.model(self.feats, self.adj.to_dense())
            if self.num_targets > 1:
                p_y = F.log_softmax(p_y, dim=1)
                q_y = F.log_softmax(q_y, dim=1)
            mask = torch.zeros(self.n_nodes, dtype=bool)
            mask[self.train_mask] = 1
            loss_train = self.criterion(self.labels, mask, p_y, q_y)
            loss_train.backward()
            self.optimizer.step()
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), q_y[self.train_mask].detach().cpu().numpy())
            loss_val, acc_val, adj_p, adj_q = self.evaluate(self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            if flag:
                self.total_time = time.time() - self.start_time
                self.result['train'] = acc_train
                self.result['valid'] = acc_val
                self.weights = deepcopy(self.model.state_dict())
                self.adjs['p'] = adj_p.detach().clone()
                self.adjs['q'] = adj_q.detach().clone()
                improve = '*'

            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.adjs

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
            p_y, adj_p, q_y, adj_q = self.model(self.feats, self.adj.to_dense())
        if self.num_targets > 1:
            p_y = F.log_softmax(p_y, dim=1)
            q_y = F.log_softmax(q_y, dim=1)
        mask = torch.zeros(self.n_nodes, dtype=bool)
        mask[val_mask] = 1
        loss = self.criterion(self.labels, mask, p_y, q_y)
        acc = self.metric(self.labels[val_mask].cpu().numpy(), q_y[val_mask].detach().cpu().numpy())
        return loss, acc, adj_p, adj_q

    def set_method(self):
        self.model = WSGNN(self.conf.model['graph_skip_conn'], self.conf.model['n_hidden'], self.conf.model['dropout'], self.conf.model['n_layers'],
                           self.conf.model['graph_learn_num_pers'], self.conf.model['mlp_layers'], self.conf.model['no_bn'], self.dim_feats,self.n_nodes,self.num_targets, self.conf).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.criterion = ELBONCLoss(binary=(self.num_targets==1))