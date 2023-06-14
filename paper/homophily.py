import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora',
                    choices=['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho',
                             'coauthorcs', 'coauthorph', 'amazon-ratings', 'questions', 'chameleon-filtered',
                             'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'penn94',
                             'blogcatalog', 'flickr'], help='dataset')
parser.add_argument('--method', type=str, default='gcn', choices=['gcn', 'appnp', 'gt', 'gat', 'prognn', 'gen',
                                                                  'gaug', 'idgl', 'grcn', 'sgc', 'jknet', 'slaps',
                                                                  'gprgnn', 'nodeformer', 'segsl', 'sublime',
                                                                  'stable', 'cogsl', 'lpa', 'link', 'linkx'], help="Select methods")
args = parser.parse_args()

import opengsl as opengsl
import numpy as np
import torch

conf = opengsl.load_conf(method=args.method, dataset=args.data)
print(conf)
data = opengsl.data.Dataset(args.data, feat_norm=conf.dataset['feat_norm'], path='data')

fill = None
h = []
print(opengsl.utils.get_homophily(data.labels.cpu(), data.adj.to_dense().cpu(), type='edge', fill=fill))
for i in range(10):
    adj = torch.load(os.path.join('results/graph/{}'.format(args.method), '{}_{}_{}.pth'.format(args.data, 0, i)))
    h.append(opengsl.utils.get_homophily(data.labels.cpu(), adj.cpu(), type='edge', fill=fill))
    print(h)
h = np.array(h)
print(f'{h.mean():.4f} ± {h.std():.4f}')