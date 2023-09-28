import torch
from copy import deepcopy
import time
from opengsl.utils.utils import accuracy
from opengsl.utils.recorder import Recorder
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
import torch.nn.functional as F


class Solver:
    '''
    Base solver class to conduct a single experiment. It defines the abstract training procedures
    which can be overwritten in subclass solver for each method.

    Parameters
    ----------
    conf : argparse.Namespace
        Configuration file.
    dataset : opengsl.data.Dataset
        Dataset to be conduct an experiment on.

    Attributes
    ----------
    conf : argparse.Namespace
        Configuration file.
    dataset : opengsl.data.Dataset
        Dataset to be conduct an experiment on.
    model : nn.Module
        Model of the method.
    loss_fn : function
        Loss function, either `F.binary_cross_entropy_with_logits` or `F.cross_entropy`.
    metric : functrion
        Metric function, either 'roc_auc_score' or 'accuracy'.

    '''
    def __init__(self, conf, dataset):
        self.dataset = dataset
        
        self.conf = conf
        self.device = torch.device('cuda')
        self.n_nodes = dataset.n_nodes
        self.dim_feats = dataset.dim_feats
        self.num_targets = dataset.num_targets
        self.n_classes = dataset.n_classes
        self.model = None
        self.loss_fn = F.binary_cross_entropy_with_logits if self.num_targets == 1 else F.cross_entropy
        self.metric = roc_auc_score if self.num_targets == 1 else accuracy
        if self.n_classes == 1:        
            self.loss_fn = torch.nn.MSELoss()
            self.metric = r2_score
        self.model = None

        self.feats = dataset.feats
        self.adj = dataset.adj if self.conf.dataset['sparse'] else dataset.adj.to_dense()
        self.labels = dataset.labels
        self.train_masks = dataset.train_masks
        self.val_masks = dataset.val_masks
        self.test_masks = dataset.test_masks

    def run_exp(self, split=None, debug=False):
        '''
        Function to start an experiment.

        Parameters
        ----------
        split : int
            Specify the idx of a split among mutiple splits. Set to 0 if not specified.
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        graph : torch.tensor
            The learned structure. `None` for GNN methods.
        '''
        self.set(split)
        return self.learn(debug)

    def set(self, split):
        '''
        This conducts necessary operations for an experiment, including the setting specified split,
        variables to record statistics, models.

        Parameters
        ----------
        split : int
            Specify the idx of a split among mutiple splits. Set to 0 if not specified.

        '''
        if split is None:
            print('split set to default 0.')
            split=0
        assert split<len(self.train_masks), 'error, split id is larger than number of splits'
        self.train_mask = self.train_masks[split]
        self.val_mask = self.val_masks[split]
        self.test_mask = self.test_masks[split]
        self.total_time = 0
        self.best_val_loss = 1e15
        self.weights = None
        self.best_graph = None
        self.result = {'train': -1, 'valid': -1, 'test': -1}
        self.start_time = time.time()
        self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
        self.adjs = {'ori':self.adj, 'final':None}
        self.set_method()

    def set_method(self):
        '''
        This sets model and other members, which is overwritten for each method.

        '''
        self.model = None
        self.optim = None

    def learn(self, debug=False):
        '''
        This is the common learning procedure, which is overwritten for special learning procedure.

        Parameters
        ----------
        debug : bool
            Whether to print statistics during training.

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        graph : torch.tensor
            The learned structure. `None` for GNN methods.
        '''

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output = self.model(self.input_distributer())
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            # save
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break


            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, None

    def evaluate(self, val_mask):
        '''
        This is the common evaluation procedure, which is overwritten for special evaluation procedure.

        Parameters
        ----------
        val_mask : torch.tensor

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_distributer())
        logits = output[val_mask]
        labels = self.labels[val_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def input_distributer(self):
        '''
        This distributes different input in `learn` for different methods, which is overwritten for each method.
        '''
        return None

    def test(self):
        '''
        This is the common test procedure, which is overwritten for special test procedure.

        Returns
        -------
        loss : float
            Test loss.
        metric : float
            Test metric.
        '''
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)





