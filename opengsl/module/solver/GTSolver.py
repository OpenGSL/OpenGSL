from copy import deepcopy
from opengsl.module.model.gt import GT
import torch
import time
from .solver import Solver
import dgl


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
        return self.evaluate(self.test_mask)

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