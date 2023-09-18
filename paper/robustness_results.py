import argparse
import os
import sys
import pandas as pd
import scipy.sparse as sp


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora',
                    choices=['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho',
                             'coauthorcs', 'coauthorph', 'amazon-ratings', 'questions', 'chameleon-filtered',
                             'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'penn94',
                             'blogcatalog', 'flickr', 'wikics', 'ogbn-arxiv', 'csbm20', 'csbm40', 'csbm60', 'csbm80'], help='dataset')
parser.add_argument('--method', type=str, default='gcn', choices=['gcn', 'appnp', 'gt', 'gat', 'prognn', 'gen',
                                                                  'gaug', 'idgl', 'grcn', 'sgc', 'jknet', 'slaps',
                                                                  'gprgnn', 'nodeformer', 'segsl', 'sublime',
                                                                  'stable', 'cogsl', 'lpa', 'link', 'linkx', 'wsgnn', 'gin'], help="Select methods")
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
parser.add_argument('--n_splits', type=int, default=1)
parser.add_argument('--n_runs', type=int, default=10)
parser.add_argument('--rate', type=float, default=0.1)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from opengsl.config import load_conf
from opengsl.data import Dataset
from opengsl import ExpManager
from opengsl.method import *
from opengsl.utils.utils import scipy_sparse_to_sparse_tensor
import torch

if args.config is None:
    conf = load_conf(method=args.method, dataset=args.data)
else:
    conf = load_conf(args.config)
conf.analysis['save_graph'] = False
print(conf)

dataset = Dataset(args.data, feat_norm=conf.dataset['feat_norm'], path='data', n_splits=args.n_splits)
# load perturbed adj
perturbed_adj = sp.load_npz(os.path.join('data/robustness', args.data+'_{}.npz'.format(args.rate)))
perturbed_adj = scipy_sparse_to_sparse_tensor(perturbed_adj).to(torch.device('cuda'))
dataset.adj = perturbed_adj


method = eval('{}Solver(conf, dataset)'.format(args.method.upper()))
exp = ExpManager(method,  save_path='records')
acc_save, std_save = exp.run(n_runs=args.n_runs, n_splits=args.n_splits, debug=args.debug)
text = '{:.2f} ± {:.2f}'.format(acc_save, std_save)

if not os.path.exists('results'):
    os.makedirs('results')
if os.path.exists('results/robustness.csv'):
    records = pd.read_csv('results/robustness.csv')
    records.loc[len(records)] = {'method':args.method, 'data':args.data, 'ptb':args.rate, 'acc':text}
    records.to_csv('results/robustness.csv', index=False)
else:
    records = pd.DataFrame([[args.method, args.data, args.rate, text]], columns=['method', 'data', 'ptb', 'acc'])
    records.to_csv('results/robustness.csv', index=False)