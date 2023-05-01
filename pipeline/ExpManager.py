import torch
from utils.utils import set_seed
from utils.logger import Logger
import os
import pandas as pd
from data.dataset import Dataset
from pipeline.gnnsolver import GCNSolver, SGCSolver, MLPSolver, LINKXSolver, LINKSolver, JKNetSolver
from pipeline.gslsolver import GRCNSolver


solvers = {
    'gcn':GCNSolver,
    'jknet':JKNetSolver,
    'sgc':SGCSolver,
    'mlp':MLPSolver,
    'link':LINKSolver,
    'linkx':LINKXSolver,
    'grcn':GRCNSolver
}


class ExpManager:
    def __init__(self, conf, method='gcn', data='cora', n_splits=1, n_runs=1, save=False, debug=False, verbose=True):
        dataset = Dataset(data, feat_norm=conf.dataset['feat_norm'], verbose=verbose, n_splits=n_splits, cora_split=conf.dataset['cora_split'])
        self.conf = conf
        self.method = method
        self.data = data
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
        if 'load_graph' in self.conf.dataset and self.conf.dataset['load_graph']:
            assert 'load_graph_path' in self.conf.analysis and self.conf.dataset[
                'load_graph_path'] is not None, 'Specify the path to load graph'
            self.load_graph_path = self.conf.dataset['load_graph_path']
        assert self.save_graph_path is None or self.load_graph_path is None, 'GNN does not save graph, GSL does not load graph'
        self.debug = debug

        Solver0 = solvers[method]
        self.solver = Solver0(conf, dataset)

    def run(self):
        total_runs = self.n_runs * self.n_splits
        assert self.n_splits <= len(self.split_seeds)
        assert total_runs <= len(self.train_seeds)
        logger = Logger(runs=total_runs)
        for i in range(self.n_splits):
            for j in range(self.n_runs):
                idx = i * self.n_runs + j
                print("Exp {}/{}".format(idx, total_runs))
                set_seed(self.train_seeds[idx])

                # load graph
                if self.load_graph_path:
                    self.solver.adj = torch.load(os.path.join(self.load_graph_path,'{}_{}_{}.pth'.format(self.data, i, self.train_seeds[idx]))).to_sparse().to(self.device)

                # run an exp
                result, graph = self.solver.run_exp(split=i, debug=self.debug)
                logger.add_result(idx, result)

                # save graph
                if self.save_graph_path:
                    if not os.path.exists(self.save_graph_path):
                        os.makedirs(self.save_graph_path)
                    torch.save(graph.cpu(), os.path.join(self.save_graph_path, '{}_{}_{}.pth'.format(self.data, i, self.train_seeds[idx])))
        self.acc_save = 100 * torch.tensor(logger.results)[:,2].mean().float()
        self.std_save = 100 * torch.tensor(logger.results)[:,2].std().float()
        logger.print_statistics()
        self.save()

    def save(self):
        # save results
        if self.save_path:
            if not os.path.exists('results'):
                os.makedirs('results')
            if os.path.exists('results/performance.csv'):
                records = pd.read_csv('results/performance.csv')
                records.loc[len(records)] = {'model':self.method, 'data':self.data, 'acc':self.acc_save, 'std':self.std_save}
                records.to_csv('results/performance.csv', index=False)
            else:
                records = pd.DataFrame(
                    [[self.method, self.data, self.acc_save, self.std_save]],
                    columns=['model', 'data', 'acc', 'std'])
                records.to_csv('results/performance.csv', index=False)