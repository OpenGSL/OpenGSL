import argparse
import os
import sys
import pandas as pd
import scipy.sparse as sp
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora',
                    choices=['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho',
                             'coauthorcs', 'coauthorph', 'amazon-ratings', 'questions', 'chameleon-filtered',
                             'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'penn94',
                             'blogcatalog', 'flickr', 'wikics', 'ogbn-arxiv', 'csbm20', 'csbm40', 'csbm60', 'csbm80'], help='dataset')
parser.add_argument('--attack', type=str, default='meta', choices=['meta', 'random'], help="Select methods")
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
parser.add_argument('--rate', type=float, default=0.1)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from opengsl.config import load_conf
from opengsl.data import Dataset
from opengsl.data.preprocess.attack import metattack
from opengsl.utils.utils import sparse_tensor_to_scipy_sparse
from opengsl import ExpManager
from opengsl.method import *

dataset = Dataset(args.data, feat_norm=False, path='data', n_splits=1)
new_adj = metattack(dataset.adj, dataset.feats, dataset.labels, dataset.train_masks[0], dataset.val_masks[0],
                    dataset.test_masks[0], ptb_rate=args.rate)
indices = new_adj.nonzero().t()
values = new_adj[indices[0], indices[1]]
new_adj = sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=new_adj.shape)
if not os.path.exists('data/robustness'):
    os.makedirs('data/robustness')
sp.save_npz(os.path.join('data/robustness', args.data+'_{}.npz'.format(args.rate)), new_adj)