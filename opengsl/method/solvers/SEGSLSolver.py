import numpy as np
from copy import deepcopy
from opengsl.method.models.gcn import GCN
from opengsl.method.models.segsl import knn_maxE1, add_knn, get_weight, get_adj_matrix, PartitionTree, get_community, reshape
import torch
import time
from .solver import Solver
from opengsl.method.functional import normalize
import dgl


class SEGSLSolver(Solver):
    '''
    A solver to train, evaluate, test SEGSL in a run.

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
    >>> conf = opengsl.config.load_conf('segsl', 'cora')
    >>>
    >>> solver = SEGSLSolver(conf, dataset)
    >>> # Conduct a experiment run.
    >>> acc, new_structure = solver.run_exp(split=0, debug=True)
    '''
    def __init__(self, conf, dataset):
        super().__init__(conf, dataset)
        self.method_name = "segsl"
        print("Solver Version : [{}]".format("segsl"))

    def learn(self, debug=False):
        '''
        Learning process of SEGSL.

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
        adj = self.adj.to_dense()
        adj.fill_diagonal_(1)
        adj = adj.to_sparse()

        for iter in range(self.conf.training['n_iters']):
            logits = self.train_gcn(iter, adj, debug)
            adj = self.structure_learning(logits, adj)

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result, self.best_graph

    def train_gcn(self, iter, adj, debug=False):
        self.model = GCN(self.dim_feats, self.conf.model['n_hidden'], self.num_targets, self.conf.model['n_layers'],
                         self.conf.model['dropout'], self.conf.model['input_dropout']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

        if debug:
            print('==== Iteration {:04d} ===='.format(iter+1))
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        normalized_adj = normalize(adj, add_loop=False, sparse=True)
        for epoch in range(self.conf.training['n_epochs']):
            improve_2 = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            hidden_output, output = self.model((self.feats, normalized_adj, False))
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
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
                    self.total_time = time.time()-self.start_time
                    improve_1 = '*'
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.best_iter = iter+1
                    self.weights = deepcopy(self.model.state_dict())
                    self.best_graph = deepcopy(adj)

            # print
            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve_2))

        print('Iteration {:04d} | Time(s) {:.4f} | Loss(val):{:.4f} | Acc(val):{:.4f} | {}'.format(iter+1,time.time()-t, best_loss_val, best_acc_val, improve_1))

        return output

    def structure_learning(self, logits, adj):
        edge_index = adj.coalesce().indices().t()

        k = knn_maxE1(edge_index, logits)  # edge index对称有自环
        edge_index_2 = add_knn(k, logits, edge_index)
        weight = get_weight(logits, edge_index_2)
        adj_matrix = get_adj_matrix(self.n_nodes, edge_index_2, weight)

        code_tree = PartitionTree(adj_matrix=np.array(adj_matrix))
        code_tree.build_coding_tree(self.conf.gsl['se'])

        community, isleaf = get_community(code_tree)
        new_edge_index = reshape(community, code_tree, isleaf,
                                 self.conf.gsl['k'])
        new_edge_index_2 = reshape(community, code_tree, isleaf,
                                   self.conf.gsl['k'])
        new_edge_index = torch.cat(
            (new_edge_index.t(), new_edge_index_2.t()), dim=0)
        new_edge_index, unique_idx = torch.unique(
            new_edge_index, return_counts=True, dim=0)
        new_edge_index = new_edge_index[unique_idx != 1].t()
        add_num = int(new_edge_index.shape[1])

        new_edge_index = torch.cat(
            (new_edge_index.t(), edge_index.cpu()), dim=0)
        new_edge_index = torch.unique(new_edge_index, dim=0)
        new_edge_index = new_edge_index.t()
        new_weight = get_weight(logits, new_edge_index.t())
        _, delete_idx = torch.topk(new_weight,
                                   k=add_num,
                                   largest=False)
        delete_mask = torch.ones(
            new_edge_index.t().shape[0]).bool()
        delete_mask[delete_idx] = False
        new_edge_index = new_edge_index.t()[delete_mask].t()  # 得到新的edge_index了

        graph = dgl.graph((new_edge_index[0], new_edge_index[1]),
                          num_nodes=self.n_nodes).to(self.device)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        adj = graph.adj().to(self.device)
        return adj

    def evaluate(self, test_mask, normalized_adj):
        '''
        Evaluation procedure of SEGSL.

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
            hidden_output, output = self.model((self.feats, normalized_adj, False))
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy()), hidden_output, output

    def set_method(self):
        '''
        Function to set the model and necessary variables for each run, automatically called in function `set`.

        '''
        self.best_iter = 0

    def test(self):
        '''
        Test procedure of SEGSL.

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
        normalized_adj = normalize(self.best_graph, add_loop=False, sparse=True)
        return self.evaluate(self.test_mask, normalized_adj)