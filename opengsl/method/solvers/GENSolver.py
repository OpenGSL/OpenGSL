from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
from copy import deepcopy
from opengsl.method.models.gen import EstimateAdj as GENEstimateAdj, prob_to_adj
import torch
import torch.nn.functional as F
import time
from .solver import Solver
from opengsl.utils.utils import get_homophily
from opengsl.method.functional import normalize
from opengsl.method.transform import KNN
from opengsl.method.encoder import GCNEncoder, APPNPEncoder, GINEncoder


class GENSolver(Solver):
    '''
    A solver to train, evaluate, test GEN in a run.

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
    >>> conf = opengsl.config.load_conf('gen', 'cora')
    >>>
    >>> solver = GENSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''

    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "gen"
        print("Solver Version : [{}]".format("gen"))
        self.homophily = get_homophily(self.labels.cpu(), self.adj.to_dense().cpu(), type='node')

    def knn(self, feature):
        # Generate a knn graph for input feature matrix. Note that the graph contains self loop.
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int64)
        dist = cos(feature.detach().cpu().numpy())
        col = np.argpartition(dist, -(self.conf.gsl['k'] + 1), axis=1)[:, -(self.conf.gsl['k'] + 1):].flatten()
        adj[np.arange(self.n_nodes).repeat(self.conf.gsl['k'] + 1), col] = 1

        # this may cause difference due to different implementations
        # tmp = KNN(self.conf.gsl['k'] + 1, metric='cosine', set_value=1)
        # adj = tmp(feature, self.conf.gsl['k'] + 1)
        return adj

    def train_gcn(self, iter, adj, debug=False):
        # if iter == 1:
        #     self.result['valid'] = 0
        if debug:
            print('==== Iteration {:04d} ===='.format(iter + 1))
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        normalized_adj = normalize(adj)
        for epoch in range(self.conf.training['n_epochs']):
            improve_2 = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            hidden_output, output = self.model(self.feats, normalized_adj, return_mid=True)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val, hidden_output, output = self.evaluate(self.val_mask, normalized_adj)

            # save
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                improve_2 = '*'
                if acc_val > self.result['valid']:
                    self.total_time = time.time() - self.start_time
                    improve_1 = '*'
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.best_iter = iter + 1
                    # if iter == 0:
                    self.hidden_output = hidden_output
                    self.output = output if len(output.shape) > 1 else output.unsqueeze(1)
                    self.output = F.log_softmax(self.output, dim=1)
                    self.weights = deepcopy(self.model.state_dict())
                    self.adjs['final'] = adj.detach().clone()

            # print
            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve_2))

        print('Iteration {:04d} | Time(s) {:.4f} | Loss(val):{:.4f} | Acc(val):{:.4f} | {}'.format(iter + 1,
                                                                                                   time.time() - t,
                                                                                                   best_loss_val,
                                                                                                   best_acc_val,
                                                                                                   improve_1))

    def structure_learning(self, iter):
        t = time.time()
        self.estimator.reset_obs()
        self.estimator.update_obs(self.knn(self.feats))  # 2
        self.estimator.update_obs(self.knn(self.hidden_output))  # 3
        self.estimator.update_obs(self.knn(self.output))  # 4
        alpha, beta, O, Q, iterations = self.estimator.EM(self.output.max(1)[1].detach().cpu().numpy(),
                                                          self.conf.gsl['tolerance'])
        adj = prob_to_adj(Q, self.conf.gsl['threshold']).to(self.device).to_sparse()
        print('Iteration {:04d} | Time(s) {:.4f} | EM step {:04d}'.format(iter + 1, time.time() - t,
                                                                          self.estimator.count))
        return adj

    def learn(self, debug=False):
        '''
        Learning process of GEN.

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
        adj = self.adj

        for iter in range(self.conf.training['n_iters']):
            self.train_gcn(iter, adj, debug)
            adj = self.structure_learning(iter)

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.adjs

    def evaluate(self, test_mask, normalized_adj):
        '''
        Evaluation procedure of GEN.

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
        hidden_output : torch.tensor
            Hidden output of the model.
        output : torch.tensor
            Output of the model.
        '''
        self.model.eval()
        with torch.no_grad():
            hidden_output, output = self.model(self.feats, normalized_adj, return_mid=True)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), hidden_output, output

    def test(self):
        '''
        Test procedure of GEN.

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        hidden_output : torch.tensor
            Hidden output of the model.
        output : torch.tensor
            Output of the model.
        '''
        self.model.load_state_dict(self.weights)
        normalized_adj = normalize(self.best_graph)
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
                             weight_initializer='glorot',
                             bias_initializer='zeros').to(self.device)
        elif self.conf.model['type'] == 'appnp':
            self.model = APPNPEncoder(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                               dropout=self.conf.model['dropout'], K=self.conf.model['K'],
                               alpha=self.conf.model['alpha']).to(self.device)
        elif self.conf.model['type'] == 'gin':
            self.model = GINEncoder(self.dim_feats, self.conf.model['n_hidden'], self.num_targets,
                             self.conf.model['n_layers'], self.conf.model['mlp_layers']).to(self.device)

        self.estimator = GENEstimateAdj(self.n_classes, self.adj.to_dense(), self.train_mask, self.labels,
                                        self.homophily)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.best_iter = 0
        self.hidden_output = None
        self.output = None