import torch
from copy import deepcopy
import time
import torch_geometric.data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from opengsl.utils.utils import accuracy
from opengsl.utils.recorder import Recorder
from opengsl.module.functional import normalize
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
        self.device = torch.device('cuda') if not ('use_cpu' in conf and conf.use_cpu) else torch.device('cpu')
        self.method_name = ''
        self.single_graph = dataset.single_graph
        if self.single_graph:
            self.n_nodes = dataset.n_nodes
            self.feats = dataset.feats
            self.adj = dataset.adj if self.conf.dataset['sparse'] else dataset.adj.to_dense()
            self.labels = dataset.labels
        else:
            self.n_graphs = dataset.n_graphs
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
        self.train_masks = dataset.train_masks
        self.val_masks = dataset.val_masks
        self.test_masks = dataset.test_masks
        self.current_split = 0

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
        if ('use_deterministic' not in self.conf) or self.conf.use_deterministic:
            torch.use_deterministic_algorithms(True)
        self.set(split)
        return self.learn_nc(debug) if self.single_graph else self.learn_gc(debug)

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
            split = 0
        assert split < len(self.train_masks), 'error, split id is larger than number of splits'
        self.train_mask = self.train_masks[split]
        self.val_mask = self.val_masks[split]
        self.test_mask = self.test_masks[split]
        self.current_split = split
        self.total_time = 0
        self.best_val_loss = 1e15
        self.weights = None
        self.best_graph = None
        self.result = {'train': -1, 'valid': -1, 'test': -1}
        self.start_time = time.time()
        self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
        if self.single_graph:
            self.adjs = {'ori': self.adj, 'final': None}
        self.set_method()

    def set_method(self):
        '''
        This sets model and other members, which is overwritten for each method.

        '''
        self.model = None
        self.optim = None

    def learn_nc(self, debug=False):
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
            output = self.model(**self.input_distributer())
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

    def learn_gc(self, debug=False):
        '''
        This is the common learning procedure for graph classification, which is overwritten for special learning procedure.

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
        if isinstance(self.dataset.data_raw, Dataset):
            train_dataset = self.dataset.data_raw[self.train_mask]
            test_dataset = self.dataset.data_raw[self.test_mask]
            val_dataset = self.dataset.data_raw[self.val_mask]
        elif isinstance(self.dataset.data_raw, list):
            train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
            test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
            val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]

        train_loader = DataLoader(train_dataset, self.conf.training['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, self.conf.training['test_batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, self.conf.training['test_batch_size'], shuffle=False)

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            loss_train = 0

            # forward and backward
            preds = []
            ground_truth = []
            self.model.train()
            for data in train_loader:
                self.optim.zero_grad()
                data = data.to(self.device)
                out = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch)
                loss = self.loss_fn(out, data.y.view(-1))
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optim.step()
                loss_train += loss.item() * data.num_graphs
                pred = F.softmax(out, dim=1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_train = loss_train / len(train_loader.dataset)
            acc_train = self.metric(ground_truth, preds)

            # Evaluate
            preds = []
            ground_truth = []
            self.model.eval()
            loss_val = 0
            for data in val_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    out = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch)
                    pred = F.softmax(out, dim=1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(data.y.detach().cpu().unsqueeze(1))
                loss_val += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_val = loss_val / len(val_loader.dataset)
            acc_val = self.metric(ground_truth, preds)
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
                        epoch + 1, time.time() - t0, loss_train, acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        # test
        preds = []
        ground_truth = []
        self.model.load_state_dict(self.weights)
        self.model.eval()
        loss_test = 0
        for data in test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch)
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:, 1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
            loss_test += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
        preds = torch.vstack(preds).squeeze().numpy()
        ground_truth = torch.vstack(ground_truth).squeeze().numpy()
        loss_test = loss_test / len(test_loader.dataset)
        acc_test = self.metric(ground_truth, preds)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))
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
            output = self.model(**self.input_distributer())
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


class GSLSolver(Solver):
    def __init__(self, conf, dataset):
        super(GSLSolver, self).__init__(conf, dataset)
        self.graphlearner = None
        self.method_name = 'gsl'

    def learn_nc(self, debug=False):
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
            adj = self.graphlearner(self.feats, normalize(self.adj))
            output = self.model(self.feats, normalize(adj, add_loop=False))
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
                self.gsl_weights = deepcopy(self.graphlearner.state_dict())
                self.adjs['final'] = adj.detach().clone()
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

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            adj = self.graphlearner(self.feats, normalize(self.adj))
            output = self.model(self.feats, normalize(adj, add_loop=False))
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

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
        self.graphlearner.load_state_dict(self.gsl_weights)
        return self.evaluate(self.test_mask)

    def set_method(self):
        '''
        This sets model and other members, which is overwritten for each method.

        '''
        self.model = None
        self.optim = None
        self.graphlearner = None




