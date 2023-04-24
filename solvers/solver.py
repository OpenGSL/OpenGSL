import torch.nn as nn
import torch
import torch.nn.functional as F
from data.pyg_load import pyg_load_dataset
from data.hetero_load import hetero_load
import dgl
from data.split import get_split
from dgl.data.utils import generate_mask_tensor
from utils.utils import sample_mask, set_seed, normalize_feats, accuracy
import numpy as np
from utils.logger import Logger
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import argparse


class BaseSolver(nn.Module):
    def __init__(self, args, conf):
        super().__init__()
        self.args = args
        self.conf = conf
        self.cfg = argparse.Namespace(**vars(args), **vars(conf))
        self.device = torch.device('cuda')
        self.prepare_data(args.data)
        self.train_seeds = [i for i in range(400)]
        self.split_seeds = [0,1,2,3,4,5,6,7,8,9]

    def prepare_data(self, ds_name):
        if ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph']:
            self.data_raw = pyg_load_dataset(ds_name)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # 这个feats尚未经过归一化
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse.FloatTensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                                [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges
            self.n_classes = self.data_raw.num_classes
            if not ('data_cpu' in self.conf and self.conf['data_cpu']):
                self.feats = self.feats.to(self.device)
                self.labels = self.labels.to(self.device)
                self.adj = self.adj.to(self.device)
            # normalize features
            if self.args.not_norm_feats:
                pass
            else:
                self.feats = normalize_feats(self.feats)

        elif ds_name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc']:
            self.g = hetero_load(ds_name)
            self.adj = self.g.adj()
            if not ('data_cpu' in self.conf and self.conf['data_cpu']):
                self.g = self.g.int().to(self.device)
                self.adj = self.adj.to(self.device)
            self.feats = self.g.ndata['feat']
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.ndata['label']
            self.n_edges = self.g.number_of_edges()
            if self.args.not_norm_feats:
                pass
            else:
                self.feats = normalize_feats(self.feats)
            self.n_classes = len(self.labels.unique())

        else:
            print('dataset not implemented')
            exit(0)

        if self.args.verbose:
            print("""----Data statistics------'
                #Nodes %d
                #Edges %d
                #Classes %d""" %
                  (self.n_nodes, self.n_edges, self.n_classes))

        self.num_targets = self.n_classes
        if self.num_targets == 2:
            self.num_targets = 1
        self.loss_fn = F.binary_cross_entropy_with_logits if self.num_targets == 1 else F.cross_entropy
        self.metric = roc_auc_score if self.num_targets == 1 else accuracy


    def split_data(self, ds_name, seed):
        if ds_name in ['coauthorcs', 'coauthorph', 'amazoncom', 'amazonpho']:
            np.random.seed(seed)
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(), train_examples_per_class=20, val_examples_per_class=30)  # 默认采取20-30-rest这种划分
            self.train_mask = generate_mask_tensor(sample_mask(train_indices, self.n_nodes))
            self.val_mask = generate_mask_tensor(sample_mask(val_indices, self.n_nodes))
            self.test_mask = generate_mask_tensor(sample_mask(test_indices, self.n_nodes))
        elif ds_name in ['cora', 'citeseer', 'pubmed']:
            if 're_split' in self.conf.dataset and self.conf.dataset['re_split']:
                np.random.seed(seed)
                train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(), train_examples_per_class=20, val_size=500, test_size=1000)
                self.train_mask = generate_mask_tensor(sample_mask(train_indices, self.n_nodes))
                self.val_mask = generate_mask_tensor(sample_mask(val_indices, self.n_nodes))
                self.test_mask = generate_mask_tensor(sample_mask(test_indices, self.n_nodes))
            else:
                self.train_mask = self.g.train_mask
                self.val_mask = self.g.val_mask
                self.test_mask = self.g.test_mask
        elif ds_name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc']:
            assert seed >= 0 and seed <= 9
            self.train_mask = self.g.ndata['train_mask'][:, seed]
            self.val_mask = self.g.ndata['val_mask'][:, seed]
            self.test_mask = self.g.ndata['test_mask'][:, seed]
        else:
            print('dataset not implemented')
            exit(0)
        self.train_mask = torch.nonzero(self.train_mask, as_tuple=False).squeeze()
        self.val_mask = torch.nonzero(self.val_mask, as_tuple=False).squeeze()
        self.test_mask = torch.nonzero(self.test_mask, as_tuple=False).squeeze()

        if self.args.verbose:
            print("""----Split statistics------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (len(self.train_mask), len(self.val_mask), len(self.test_mask)))

    def run(self):
        total_runs = self.args.n_runs * self.args.n_splits
        assert self.args.n_splits <= len(self.split_seeds)
        assert total_runs <= len(self.train_seeds)
        logger = Logger(runs=total_runs)
        for i in range(self.args.n_splits):
            self.split_data(self.args.data, self.split_seeds[i])   # split the data
            # self.split_data_v2(self.split_seeds[i])
            for j in range(self.args.n_runs):
                idx = i * self.args.n_runs + j
                print("Exp {}/{}".format(idx, total_runs))
                set_seed(self.train_seeds[idx])
                if 'load_graph' in self.conf.dataset and self.conf.dataset['load_graph']['flag']:
                    self.adj = torch.load(os.path.join(self.conf.dataset['load_graph']['loc'],'{}_{}_{}.pth'.format(self.args.data, i, self.train_seeds[idx]))).to_sparse().to(self.device)
                result = self.train()
                logger.add_result(idx, result)
                if 'save_graph' in self.conf.analysis and self.conf.analysis['save_graph']:
                    self.save_graph(i, self.train_seeds[idx])
        self.acc_save = 100 * torch.tensor(logger.results)[:,2].mean().float()
        self.std_save = 100 * torch.tensor(logger.results)[:,2].std().float()
        logger.print_statistics()
        self.save()

    def save(self):
        # save results
        if self.args.save:
            if not os.path.exists('results'):
                os.makedirs('results')
            if os.path.exists('results/performance.csv'):
                records = pd.read_csv('results/performance.csv')
                records.loc[len(records)] = {'model':self.args.solver, 'data':self.args.data, 'acc':self.acc_save, 'std':self.std_save}
                records.to_csv('results/performance.csv', index=False)
            else:
                records = pd.DataFrame(
                    [[self.args.solver, self.args.data, self.acc_save, self.std_save]],
                    columns=['model', 'data', 'acc', 'std'])
                records.to_csv('results/performance.csv', index=False)

    def save_graph(self, split, seed):
        if not os.path.exists('results/graph/{}'.format(self.args.solver)):
            os.makedirs('results/graph/{}'.format(self.args.solver))
        torch.save(self.best_graph.cpu(), 'results/graph/{}/{}_{}_{}.pth'.format(self.args.solver, self.args.data, split, seed))