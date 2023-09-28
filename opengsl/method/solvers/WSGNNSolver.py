from copy import deepcopy
from opengsl.method.models.wsgnn import WSGNN, ELBONCLoss
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
                           self.conf.model['graph_learn_num_pers'], self.conf.model['mlp_layers'], self.conf.model['no_bn'], self.dim_feats,self.n_nodes,self.num_targets, self.conf).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.criterion = ELBONCLoss(binary=(self.num_targets==1))