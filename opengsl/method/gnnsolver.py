from .models.gnn_modules import SGC, LPA, MLP, LINK, LINKX, APPNP, GPRGNN, GAT
from .models.wsgnn import WSGNN, ELBONCLoss
import time
from .models.gcn import GCN
from .models.jknet import JKNet
from .solver import Solver
import torch
from copy import deepcopy
from opengsl.data.preprocess.normalize import normalize


class SGCSolver(Solver):
    '''
    A solver to train, evaluate, test SGC in a run.

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
    >>> conf = opengsl.config.load_conf('sgc', 'cora')
    >>>
    >>> solver = SGCSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''

    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "sgc"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.feats : torch.tensor
            Node features.
        self.normalized_adj : torch.tensor
            Adjacency matrix.
        '''
        return self.feats, self.normalized_adj

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = SGC(self.dim_feats, self.num_targets, self.conf.model['n_layers']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                           weight_decay=self.conf.training['weight_decay'])
        if self.conf.dataset['normalize']:
            self.normalize = normalize
        else:
            self.normalize = lambda x, y: x
        self.normalized_adj = self.normalize(self.adj, add_loop=self.conf.dataset['add_loop'], sparse=self.conf.dataset['sparse'])


class GCNSolver(Solver):
    '''
    A solver to train, evaluate, test GCN in a run.

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
    >>> conf = opengsl.config.load_conf('gcn', 'cora')
    >>>
    >>> solver = GCNSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "gcn"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.feats : torch.tensor
            Node features.
        self.normalized_adj : torch.tensor
            Adjacency matrix.
        True : constant bool
        '''
        return self.feats, self.normalized_adj, True

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                    self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                    self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                    self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        if self.conf.dataset['normalize']:
            self.normalize = normalize
        else:
            self.normalize = lambda x, y: x
        self.normalized_adj = self.normalize(self.adj, add_loop=self.conf.dataset['add_loop'], sparse=self.conf.dataset['sparse'])


class LPASolver(Solver):
    '''
    A solver to test LPA in a run.

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
    >>> conf = opengsl.config.load_conf('lpa', 'cora')
    >>>
    >>> solver = LPASolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "lpa"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.labels : torch.tensor
            Node labels.
        self.normalized_adj : torch.tensor
            Adjacency matrix.
        self.train_mask : torch.tensor
            A boolean tensor indicating whether the node is in the training set.
        '''
        return self.labels, self.normalized_adj, self.train_mask

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = LPA(self.conf.model['n_layers'], self.conf.model['alpha']).to(self.device)
        self.normalize = normalize if self.conf.dataset['normalize'] else lambda x, y: x
        self.normalized_adj = self.normalize(self.adj, add_loop=self.conf.dataset['add_loop'], sparse=self.conf.dataset['sparse'])

    def learn(self, split=None, debug=False):
        '''
        Learning process of LPA.

        Parameters
        ----------
        debug : bool

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        0 : constant

        '''
        y_pred = self.model(self.input_distributer())
        loss_test = self.loss_fn(y_pred[self.test_mask], self.labels[self.test_mask])
        acc_test = self.metric(self.labels[self.test_mask].cpu().numpy(), y_pred[self.test_mask].detach().cpu().numpy())

        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, 0


class MLPSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "mlp"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.feats : torch.tensor
            Node features.
        '''
        return self.feats

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = MLP(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])


class LINKSolver(Solver):
    '''
    A solver to train, evaluate, test LINK in a run.

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
    >>> conf = opengsl.config.load_conf('link', 'cora')
    >>>
    >>> solver = LINKSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "link"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.adj : torch.tensor
            Adjacency matrix.
        '''
        return self.adj

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = LINK(self.n_nodes, self.num_targets).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])


class LINKXSolver(Solver):
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "linkx"

    def input_distributer(self):
        return self.feats, self.adj

    def set_method(self):
        self.model = LINKX(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'], self.n_nodes).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])


class APPNPSolver(Solver):
    '''
    A solver to train, evaluate, test APPNP in a run.

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
    >>> conf = opengsl.config.load_conf('appnp', 'cora')
    >>>
    >>> solver = APPNPSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "appnp"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.feats : torch.tensor
            Node features.
        self.normalized_adj : torch.tensor
            Adjacency matrix.
        '''
        return self.feats, self.normalized_adj

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = APPNP(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, dropout=self.conf.model['dropout'],
                      K=self.conf.model['K'], alpha=self.conf.model['alpha']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                 weight_decay=self.conf.training['weight_decay'])
        if self.conf.dataset['normalize']:
            self.normalize = normalize
        else:
            self.normalize = lambda x, y: x
        self.normalized_adj = self.normalize(self.adj, self.conf.dataset['add_loop'], sparse=self.conf.dataset['sparse'])


class JKNetSolver(Solver):
    '''
    A solver to train, evaluate, test JKNet in a run.

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
    >>> conf = opengsl.config.load_conf('jknet', 'cora')
    >>>
    >>> solver = JKNetSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "jknet"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.feats : torch.tensor
            Node features.
        self.normalized_adj : torch.tensor
            Adjacency matrix.
        True : constant bool
        '''
        return self.feats, self.normalized_adj, True

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = JKNet(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                           self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                           self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                           self.conf.model['general'],
                           self.conf.model['input_layer'], self.conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        if self.conf.dataset['normalize']:
            self.normalize = normalize
        else:
            self.normalize = lambda x, y: x
        self.normalized_adj = self.normalize(self.adj, self.conf.dataset['add_loop'], self.conf.dataset['sparse'])


class GPRGNNSolver(Solver):
    '''
    A solver to train, evaluate, test GPRGNN in a run.

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
    >>> conf = opengsl.config.load_conf('gprgnn', 'cora')
    >>>
    >>> solver = GPRGNNSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "gprgnn"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.feats : torch.tensor
            Node features.
        self.normalized_adj : torch.tensor
            Adjacency matrix.
        '''
        return self.feats, self.normalized_adj

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = GPRGNN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, dropout=self.conf.model['dropout'],
                      dprate=self.conf.model['dprate'], K=self.conf.model['K'], alpha=self.conf.model['alpha'], init=self.conf.model['init']).to(self.device)
        self.optim = torch.optim.Adam([{
                'params': self.model.lin1.parameters(),
                'weight_decay': self.conf.training['weight_decay'], 'lr': self.conf.training['lr']
            }, {
                'params': self.model.lin2.parameters(),
                'weight_decay': self.conf.training['weight_decay'], 'lr': self.conf.training['lr']
            }, {
                'params': self.model.temp,
                'weight_decay': 0.0, 'lr': self.conf.training['lr']
            }], lr=self.conf.training['lr'])
        if self.conf.dataset['normalize']:
            self.normalize = normalize
        else:
            self.normalize = lambda x, y: x
        self.normalized_adj = self.normalize(self.adj, self.conf.dataset['add_loop'], self.conf.dataset['sparse'])


class GATSolver(Solver):
    '''
    A solver to train, evaluate, test GAT in a run.

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
    >>> conf = opengsl.config.load_conf('gat', 'cora')
    >>>
    >>> solver = GATSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, _ = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "gat"

    def input_distributer(self):
        '''
        Function to ditribute input to GNNs, automatically called in function `learn`.

        Returns
        -------
        self.feats : torch.tensor
            Node features.
        self.edge_index : torch.tensor
            Adjacency matrix.
        '''
        return self.feats, self.edge_index

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.model = GAT(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                         n_heads=self.conf.model['n_heads'], dropout=self.conf.model['dropout']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                           weight_decay=self.conf.training['weight_decay'])
        # prepare edge index
        self.edge_index = self.adj.coalesce().indices()


class WSGNNSolver(Solver):
    '''
    A solver to train, evaluate, test CoGSL in a run.

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
        p_y = torch.nn.functional.log_softmax(p_y, dim=1)
        q_y = torch.nn.functional.log_softmax(q_y, dim=1)
        mask = torch.zeros(self.n_nodes, dtype=bool)
        mask[val_mask] = 1
        loss = self.criterion(self.labels, mask, p_y, q_y)
        acc = self.metric(self.labels[val_mask].cpu().numpy(), q_y[val_mask].detach().cpu().numpy())
        return loss, acc

    def set_method(self):
        self.model = WSGNN(self.conf.model['graph_skip_conn'], self.conf.model['n_hidden'], self.conf.model['dropout'],self.conf.model['hops'],
                           self.conf.model['alpha'], self.conf.model['graph_learn_num_pers'], self.conf.model['mlp_layers'], self.conf.model['no_bn'], self.dim_feats,self.n_nodes,self.num_targets).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.criterion = ELBONCLoss()