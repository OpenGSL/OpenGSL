import torch
from utils.utils import set_seed
from utils.logger import Logger
import os
import pandas as pd
from data.dataset import Dataset
from pipeline.gnnsolver import GCNSolver, SGCSolver, MLPSolver, LINKXSolver, LINKSolver, APPNPSolver, JKNetSolver, \
    GPRGNNSolver, LPASolver
from pipeline.gslsolver import GRCNSolver, GAUGSolver, GENSolver, IDGLSolver, PROGNNSolver, GTSolver, SLAPSSolver, \
    NODEFORMERSolver, SEGSLSolver, GSRSolver, SUBLIMESolver, STABLESolver, CoGSLSolver
import argparse
import wandb


solvers = {
    'gcn':GCNSolver,
    'sgc':SGCSolver,
    'mlp':MLPSolver,
    'lpa': LPASolver,
    'link':LINKSolver,
    'linkx':LINKXSolver,
    'appnp':APPNPSolver,
    'jknet':JKNetSolver,
    'grcn':GRCNSolver,
    'gaug':GAUGSolver,
    'gen':GENSolver,
    'idgl':IDGLSolver,
    'prognn':PROGNNSolver,
    'gt':GTSolver,
    'slaps':SLAPSSolver,
    'gprgnn':GPRGNNSolver,
    'nodeformer': NODEFORMERSolver,
    'segsl': SEGSLSolver,
    'gsr':GSRSolver,
    'sublime': SUBLIMESolver,
    'stable': STABLESolver,
    'cogsl':CoGSLSolver
}


class ExpManager:
    def __init__(self, conf, method='gcn', data='cora', n_splits=1, n_runs=1, save=False, debug=False, verbose=True):
        homophily_control = None
        if 'homophily_control' in conf.dataset:
            homophily_control = conf.dataset['homophily_control']
        dataset = Dataset(data, feat_norm=conf.dataset['feat_norm'], verbose=verbose, n_splits=n_splits, cora_split=conf.dataset['cora_split'], homophily_control=homophily_control)
        # self.conf = conf
        self.conf = argparse.Namespace(**vars(conf), **{'data': data, 'method': method})
        if 'sweep' in self.conf.analysis and self.conf.analysis['sweep']:
            self.conf.analysis['flag'] = True
            assert self.conf.analysis['sweep_id'] is not None, 'Specify the sweep id'
            assert n_runs == 1, 'Sweep only supports 1 run'
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
        if 'load_graph' in self.conf.analysis and self.conf.analysis['load_graph']:
            assert 'load_graph_path' in self.conf.analysis and self.conf.analysis[
                'load_graph_path'] is not None, 'Specify the path to load graph'
            self.load_graph_path = self.conf.analysis['load_graph_path']
        assert self.save_graph_path is None or self.load_graph_path is None, 'GNN does not save graph, GSL does not load graph'
        self.debug = debug

        Solver0 = solvers[method]
        self.solver = Solver0(self.conf, dataset)

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
                    # print(self.solver.adj)
                    # exit(0)

                # run an exp
                result, graph = self.solver.run_exp(split=i, debug=self.debug)
                logger.add_result(idx, result)

                # save graph
                if self.save_graph_path:
                    if not os.path.exists(self.save_graph_path):
                        os.makedirs(self.save_graph_path)
                    torch.save(graph.cpu(), os.path.join(self.save_graph_path, '{}_{}_{}.pth'.format(self.data, i, self.train_seeds[idx])))
        self.acc_save, self.std_save = logger.print_statistics()
        self.save()

    def sweep(self):
        def one_sweep():
            wandb.init()
            # print(self.conf)
            # print(wandb.config)
            self.update_conf(wandb.config)
            # print(self.conf)
            set_seed(42)
            if self.load_graph_path:
                self.solver.adj = torch.load(os.path.join(self.load_graph_path, '{}_{}_{}.pth'.format(
                    self.data, 0, 0))).to_sparse().to(self.device)
            result, graph = self.solver.run_exp(split=0, debug=self.debug)
            acc_val = result['valid']
            wandb.log({'acc_val_max': acc_val})
            wandb.finish()

        wandb.agent(self.conf.analysis['sweep_id'], function=one_sweep, count=self.conf.analysis['count'])



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

    def update_conf(self, config):
        '''
        Set the config file according to wandb controler in sweep mode.
        Returns
        -------
        '''
        conf = vars(self.conf)
        for k, v in conf.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk in config.keys():
                        v[kk] = config[kk]
            else:
                if k in config.keys():
                    conf[k] = config[k]
        self.conf = argparse.Namespace(**conf)