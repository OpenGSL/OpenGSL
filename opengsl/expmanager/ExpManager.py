import torch
from opengsl.utils.utils import set_seed
from opengsl.utils.logger import Logger
from opengsl.config.util import save_conf
import os
import time as time
import copy
import ruamel.yaml as yaml
from torch_geometric import seed_everything


class ExpManager:
    '''
    Experiment management class to enable running multiple experiment,
    loading learned structures and saving results.

    Parameters
    ----------
    solver : opengsl.method.Solver
        Solver of the method to solve the task.
    save_path : str
        Path to save the config file.

    Examples
    --------
    >>> # load dataset
    >>> import opengsl.dataset
    >>> dataset = opengsl.dataset.Dataset('cora', feat_norm=True)
    >>> # load config file
    >>> import opengsl.config.load_conf
    >>> conf = opengsl.config.load_conf('gcn', 'cora')
    >>> # create solver
    >>> import opengsl.method.SGCSolver
    >>> solver = SGCSolver(conf, dataset)
    >>>
    >>> import opengsl.ExpManager
    >>> exp = ExpManager(solver)
    >>> exp.run(n_runs=10, debug=True)

    '''
    def __init__(self, solver=None, save_path='records'):
        self.solver = solver
        self.conf = solver.conf
        self.method = solver.method_name
        self.dataset = solver.dataset
        self.data = self.dataset.name
        self.device = torch.device('cuda')
        # you can change random seed here
        self.train_seeds = [i for i in range(400)]
        self.save_path = None
        self.save_graph_path = None
        self.load_graph_path = None
        self.logger = Logger()
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_path = save_path
        if 'analysis' in self.conf:
            if 'save_graph' in self.conf.analysis and self.conf.analysis['save_graph']:
                assert 'save_graph_path' in self.conf.analysis and self.conf.analysis['save_graph_path'] is not None, 'Specify the path to save graph'
                self.save_graph_path = os.path.join(self.conf.analysis['save_graph_path'], self.method)
            if 'load_graph' in self.conf.analysis and self.conf.analysis['load_graph']:
                assert 'load_graph_path' in self.conf.analysis and self.conf.analysis[
                    'load_graph_path'] is not None, 'Specify the path to load graph'
                self.load_graph_path = self.conf.analysis['load_graph_path']
        assert self.save_graph_path is None or self.load_graph_path is None, 'GNN does not save graph, GSL does not load graph'

    def run(self, n_splits=1, n_runs=1, debug=False):
        '''
        Run experiments with specified solver for repeated times.
        Parameters
        ----------
        n_splits : int
            Number of data splits to run experiments on.
        n_runs : int
            Number of experiment runs each split.
        debug : bool
            Whether to print statistics during training.
        Returns
        -------
        acc : float
            Mean Accuracy.
        std : float
            Standard Deviation.

        '''
        if n_splits is None:
            n_splits = self.solver.dataset.total_splits
        total_runs = n_runs * n_splits
        assert total_runs <= len(self.train_seeds)
        succeed = 0
        for i in range(n_splits):
            for j in range(400):
                idx = i * n_runs + j
                print("Exp {}/{}".format(idx, total_runs))
                seed_everything(self.train_seeds[j])

                # load graph
                if self.load_graph_path:
                    self.solver.adj = torch.load(os.path.join(self.load_graph_path,'{}_{}_{}.pth'.format(self.data, i, self.train_seeds[idx]))).to(self.device)
                    if self.conf.dataset['sparse']:
                        self.solver.adj = self.solver.adj.to_sparse()
                # run an exp
                # try:
                result, graph = self.solver.run_exp(split=i, debug=debug)
                # except ValueError:
                #     continue
                self.logger.add_result(succeed, result)

                # save graph
                if self.save_graph_path:
                    if not os.path.exists(self.save_graph_path):
                        os.makedirs(self.save_graph_path)
                    torch.save(graph.cpu(), os.path.join(self.save_graph_path, '{}_{}_{}.pth'.format(self.data, i, self.train_seeds[succeed])))
                succeed += 1
                if succeed % n_runs == 0:
                    break
        self.acc_save, self.std_save = self.logger.print_statistics()
        self.save(self.logger.agg_stats)
        return float(self.acc_save), float(self.std_save)

    def save(self, stats):
        d = {}
        d['result'] = stats
        d.update(vars(copy.deepcopy(self.conf)))
        path = os.path.join(self.save_path, '{}_{}_'.format(self.method, self.data)+time.strftime('%Y%m%d_%H%M%S', time.localtime())+'.yaml')
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, indent=2)
