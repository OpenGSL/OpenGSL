import torch
from opengsl.utils.utils import set_seed
from opengsl.utils.logger import Logger
from opengsl.config.util import save_conf
import os
import time as time


class ExpManager:
    '''
    Experiment management class to enable running multiple experiment,
    loading learned structures and saving results.

    Parameters
    ----------
    solver : opengsl.method.Solver
        Solver of the method to solve the task.
    n_splits : int
        Number of data splits to run experiment on.
    n_runs : int
        Number of experiment runs each split.
    save_path : str
        Path to save the config file.
    debug : bool
        Whether to print statistics during training.

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
    def __init__(self, solver=None, save_path=None):
        self.solver = solver
        self.conf = solver.conf
        self.method = solver.method_name
        self.dataset = solver.dataset
        self.data = self.dataset.name
        self.device = torch.device('cuda')
        # you can change random seed here
        self.train_seeds = [i for i in range(400)]
        self.split_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.save_path = None
        self.save_graph_path = None
        self.load_graph_path = None
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_path = save_path
        if 'save_graph' in self.conf.analysis and self.conf.analysis['save_graph']:
            assert 'save_graph_path' in self.conf.analysis and self.conf.analysis['save_graph_path'] is not None, 'Specify the path to save graph'
            self.save_graph_path = os.path.join(self.conf.analysis['save_graph_path'], self.method)
        if 'load_graph' in self.conf.analysis and self.conf.analysis['load_graph']:
            assert 'load_graph_path' in self.conf.analysis and self.conf.analysis[
                'load_graph_path'] is not None, 'Specify the path to load graph'
            self.load_graph_path = self.conf.analysis['load_graph_path']
        assert self.save_graph_path is None or self.load_graph_path is None, 'GNN does not save graph, GSL does not load graph'

    def run(self, n_splits=1, n_runs=1, debug=False):
        total_runs = n_runs * n_splits
        assert n_splits <= len(self.split_seeds)
        assert total_runs <= len(self.train_seeds)
        logger = Logger(runs=total_runs)
        for i in range(n_splits):
            succeed = 0
            for j in range(400):
                idx = i * n_runs + j
                print("Exp {}/{}".format(idx, total_runs))
                set_seed(self.train_seeds[idx])

                # load graph
                if self.load_graph_path:
                    self.solver.adj = torch.load(os.path.join(self.load_graph_path,'{}_{}_{}.pth'.format(self.data, i, self.train_seeds[idx]))).to(self.device)
                    if self.conf.dataset['sparse']:
                        self.solver.adj = self.solver.adj.to_sparse()
                # run an exp
                try:
                    result, graph = self.solver.run_exp(split=i, debug=debug)
                except ValueError:
                    continue
                logger.add_result(succeed, result)

                # save graph
                if self.save_graph_path:
                    if not os.path.exists(self.save_graph_path):
                        os.makedirs(self.save_graph_path)
                    torch.save(graph.cpu(), os.path.join(self.save_graph_path, '{}_{}_{}.pth'.format(self.data, i, self.train_seeds[succeed])))
                succeed += 1
                if succeed == total_runs:
                    break
        self.acc_save, self.std_save = logger.print_statistics()
        if self.save_path:
            save_conf(os.path.join(self.save_path, '{}-{}-'.format(self.method, self.data) +
                                   time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.yaml'), self.conf)
            
        return float(self.acc_save), float(self.std_save)
