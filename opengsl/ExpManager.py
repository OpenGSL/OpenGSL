import torch
from .utils.utils import set_seed
from .utils.logger import Logger
import os
import pandas as pd

import argparse


class ExpManager:
    def __init__(self, solver=None, n_splits=1, n_runs=1, save=False, debug=False, verbose=True):
        self.solver = solver
        self.conf = solver.conf
        self.method = solver.method_name
        self.dataset = solver.dataset
        self.data = self.dataset.name
        self.device = torch.device('cuda')
        # you can change random seed here
        self.train_seeds = [i for i in range(400)]
        self.split_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.n_splits = n_splits
        self.n_runs = n_runs
        self.save_path = None
        self.save_graph_path = None
        self.load_graph_path = None
        if save:
            self.save_path = 'results/performance.csv'
        if 'save_graph' in self.conf.analysis and self.conf.analysis['save_graph']:
            assert 'save_graph_path' in self.conf.analysis and self.conf.analysis['save_graph_path'] is not None, 'Specify the path to save graph'
            self.save_graph_path = os.path.join(self.conf.analysis['save_graph_path'], self.method)
        if 'load_graph' in self.conf.analysis and self.conf.analysis['load_graph']:
            assert 'load_graph_path' in self.conf.analysis and self.conf.analysis[
                'load_graph_path'] is not None, 'Specify the path to load graph'
            self.load_graph_path = self.conf.analysis['load_graph_path']
        assert self.save_graph_path is None or self.load_graph_path is None, 'GNN does not save graph, GSL does not load graph'
        self.debug = debug

    def run(self):
        total_runs = self.n_runs * self.n_splits
        assert self.n_splits <= len(self.split_seeds)
        assert total_runs <= len(self.train_seeds)
        logger = Logger(runs=total_runs)
        for i in range(self.n_splits):
            succeed = 0
            for j in range(400):
                idx = i * self.n_runs + j
                print("Exp {}/{}".format(idx, total_runs))
                set_seed(self.train_seeds[idx])

                # load graph
                if self.load_graph_path:
                    self.solver.adj = torch.load(os.path.join(self.load_graph_path,'{}_{}_{}.pth'.format(self.data, i, self.train_seeds[idx]))).to(self.device)
                    if self.conf.dataset['sparse']:
                        self.solver.adj = self.solver.adj.to_sparse()

                # run an exp
                try:
                    result, graph = self.solver.run_exp(split=i, debug=self.debug)
                except ValueError:
                    continue
                logger.add_result(succeed, result)

                # save graph
                if self.save_graph_path:
                    if not os.path.exists(self.save_graph_path):
                        os.makedirs(self.save_graph_path)
                    torch.save(graph.cpu(), os.path.join(self.save_graph_path, '{}_{}_{}.pth'.format(self.data, i, self.train_seeds[succeed])))
                succeed += 1
                if succeed == self.n_runs:
                    break
        self.acc_save, self.std_save = logger.print_statistics()
        self.save()
   
    def save(self):
        # save results
        if self.save_path:
            text = '{:.2f} ± {:.2f}'.format(self.acc_save, self.std_save)
            if not os.path.exists('results'):
                os.makedirs('results')
            if os.path.exists('results/performance.csv'):
                records = pd.read_csv('results/performance.csv')
                records.loc[len(records)] = {'model':self.method, 'data':self.data, 'acc':text}
                records.to_csv('results/performance.csv', index=False)
            else:
                records = pd.DataFrame(
                    [[self.method, self.data, text]],
                    columns=['model', 'data', 'acc'])
                records.to_csv('results/performance.csv', index=False)