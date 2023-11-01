from copy import deepcopy
from opengsl.module.model.nodeformer import NodeFormer, adj_mul
import torch
import time
from .solver import Solver


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
        self.adjs_ = []
        self.adjs_.append(adj)
        for i in range(conf.model['rb_order'] - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, self.n_nodes)
            self.adjs_.append(adj)

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
            output, link_loss = self.model(self.feats, self.adjs_, self.conf.model['tau'])
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
            output, _ = self.model(self.feats, self.adjs_, self.conf.model['tau'])
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())