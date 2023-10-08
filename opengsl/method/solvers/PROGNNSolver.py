from copy import deepcopy
from .solver import Solver
from opengsl.method.encoder import GCNEncoder, APPNPEncoder, GINEncoder
from opengsl.method.regularizer import norm_regularizer, smoothness_regularizer, PGD, ProxOperators
from opengsl.method.functional import normalize, symmetry
from opengsl.method.graphlearner import FGPLearner
import torch
import time


class PROGNNSolver(Solver):
    '''
    A solver to train, evaluate, test ProGNN in a run.

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
    >>> conf = opengsl.config.load_conf('prognn', 'cora')
    >>>
    >>> solver = PROGNNSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "prognn"
        print("Solver Version : [{}]".format("prognn"))
        self.adj = self.adj.to_dense()

    def train_gcn(self, epoch, debug=False):
        normalized_adj = symmetry(self.estimator.Adj) if self.conf.gsl['symmetric'] else self.estimator.Adj
        normalized_adj = normalize(normalized_adj)

        t = time.time()
        improve = ''
        self.model.train()
        self.optimizer.zero_grad()

        # forward and backward
        output = self.model(self.feats, normalized_adj)
        loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
        acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
        loss_train.backward()
        self.optimizer.step()

        # evaluate
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)
        flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

        # save best model
        if flag:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.adjs['final'] = self.estimator.Adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        if debug:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                epoch+1, time.time() -t, loss_train.item(), acc_train, loss_val, acc_val, improve))

    def train_adj(self, epoch, debug=False):
        estimator = self.estimator
        t = time.time()
        improve = ''
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = norm_regularizer(estimator.Adj, 1)
        loss_fro = norm_regularizer(estimator.Adj - self.adj, p='fro')
        normalized_adj = symmetry(estimator.Adj) if self.conf.gsl['symmetric'] else estimator.Adj
        normalized_adj = normalize(normalized_adj)

        if self.conf.gsl['lambda_']:
            loss_smooth_feat = smoothness_regularizer(self.feats, estimator.Adj, style='symmetric', symmetric=True)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(self.feats, normalized_adj)
        loss_gcn = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
        acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())

        #loss_symmetric = torch.norm(estimator.Adj - estimator.Adj.t(), p="fro")
        #loss_differential =  loss_fro + self.conf.gamma * loss_gcn + self.conf.lambda_ * loss_smooth_feat + args.phi * loss_symmetric
        loss_differential = loss_fro + self.conf.gsl['gamma'] * loss_gcn + self.conf.gsl['lambda_'] * loss_smooth_feat
        loss_differential.backward()
        self.optimizer_adj.step()
        # we finish the optimization of the differential part above, next we need to do the optimization of loss_l1 and loss_nuclear

        loss_nuclear =  0 * loss_fro
        if self.conf.gsl['beta'] != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = self.prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                     + self.conf.gsl['gamma'] * loss_gcn \
                     + self.conf.gsl['alpha'] * loss_l1 \
                     + self.conf.gsl['beta'] * loss_nuclear
                     #+ self.conf.phi * loss_symmetric

        estimator.Adj.data.copy_(torch.clamp(estimator.Adj.data, min=0, max=1))

        # evaluate
        self.model.eval()
        normalized_adj = symmetry(estimator.Adj) if self.conf.gsl['symmetric'] else estimator.Adj
        normalized_adj = normalize(normalized_adj)
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)
        flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

        # save the best model
        if flag:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.adjs['final'] = estimator.Adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        if debug:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss(adj) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                epoch+1, time.time() - t, total_loss.item(), loss_val, acc_val, improve))

    def learn(self, debug=False):
        '''
        Learning process of PROGNN.

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
        for epoch in range(self.conf.training['n_epochs']):
            for i in range(int(self.conf.training['outer_steps'])):
                self.train_adj(epoch, debug=debug)

            for i in range(int(self.conf.training['inner_steps'])):
                self.train_gcn(epoch, debug=debug)

            # we use earlystopping here as prognn is very slow
            if self.improve:
                self.wait = 0
                self.improve = False
            else:
                self.wait += 1
                if self.wait == self.conf.training['patience_iter']:
                    print('Early stop!')
                    break

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.adjs

    def evaluate(self, test_mask, normalized_adj):
        '''
        Evaluation procedure of PROGNN.

        Parameters
        ----------
        test_mask : torch.tensor
            A boolean tensor indicating whether the node is in the data set.
        normalized_adj : torch.tensor
            Adjacency matrix.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        self.estimator.eval()
        with torch.no_grad():
            logits = self.model(self.feats, normalized_adj)
        logits = logits[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        '''
        Test procedure of PROGNN.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.load_state_dict(self.weights)
        normalized_adj = symmetry(self.adjs['final']) if self.conf.gsl['symmetric'] else self.adjs['final']
        normalized_adj = normalize(normalized_adj)
        return self.evaluate(self.test_mask, normalized_adj)

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.
        '''
        if self.conf.model['type'] == 'gcn':
            self.model = GCNEncoder(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                             self.conf.model['dropout'], self.conf.model['input_dropout'], self.conf.model['norm'],
                             self.conf.model['n_linear'], self.conf.model['spmm_type'], self.conf.model['act'],
                             self.conf.model['input_layer'], self.conf.model['output_layer'],
                             weight_initializer='uniform').to(self.device)
        elif self.conf.model['type'] == 'appnp':
            self.model = APPNPEncoder(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                               dropout=self.conf.model['dropout'], K=self.conf.model['K'],
                               alpha=self.conf.model['alpha']).to(self.device)
        elif self.conf.model['type'] == 'gin':
            self.model = GINEncoder(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                               self.conf.model['n_layers'], self.conf.model['mlp_layers']).to(self.device)
        # self.estimator = EstimateAdj(self.adj, symmetric=self.conf.gsl['symmetric'], device=self.device).to(self.device)
        self.estimator = FGPLearner(self.adj.shape[0], nonlinear='lambda x: x').to(self.device)
        self.estimator.init_estimation(self.adj)
        self.prox_operators = ProxOperators()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                          weight_decay=self.conf.training['weight_decay'])
        self.optimizer_adj = torch.optim.SGD(self.estimator.parameters(), momentum=0.9, lr=self.conf.training['lr_adj'])
        self.optimizer_l1 = PGD(self.estimator.parameters(), proxs=[self.prox_operators.prox_l1],
                                lr=self.conf.training['lr_adj'], alphas=[self.conf.gsl['alpha']])
        self.optimizer_nuclear = PGD(self.estimator.parameters(), proxs=[self.prox_operators.prox_nuclear_cuda],
                                     lr=self.conf.training['lr_adj'], alphas=[self.conf.gsl['beta']])

        self.wait = 0