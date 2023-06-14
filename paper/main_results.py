import argparse
import os


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
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import opengsl as opengsl

conf = opengsl.load_conf(method=args.method, dataset=args.data)
if not args.method == 'gcn':
    conf.analysis['save_graph'] = True
    conf.analysis['save_graph_path'] = 'results/graph'
print(conf)
dataset = opengsl.data.Dataset(args.data, feat_norm=conf.dataset['feat_norm'], path='data')


method = eval('opengsl.method.{}(conf, dataset)'.format(args.method))
exp = opengsl.ExpManager(method, n_runs=10, debug=args.debug)
exp.run()