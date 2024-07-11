from copy import deepcopy
from opengsl.module.model.slaps import SLAPS
import torch
import time
from .solver import Solver


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
        >>> dataset = opengsl.data.Dataset('cora', feat_norm=False)
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
        self.model = SLAPS(self.n_nodes, self.dim_feats, self.num_targets, self.feats, self.device, self.conf).to(
            self.device)

    def learn_nc(self, debug=False):
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
        for epoch in range(1, self.conf.training['n_epochs'] + 1):
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
                loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask]) + self.conf.training[
                    'lamda'] * loss_dae
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
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
        self.model.reset_parameters()
        self.optim = torch.optim.Adam([
            {'params': self.model.gcn_c.parameters(), 'lr': self.conf.training['lr'],
             'weight_decay': self.conf.training['weight_decay']},
            {'params': self.model.gcn_dae.parameters(), 'lr': self.conf.training['lr_dae'],
             'weight_decay': self.conf.training['weight_decay_dae']}
        ])