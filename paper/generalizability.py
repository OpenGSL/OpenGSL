import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora',
                    choices=['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho',
                             'coauthorcs', 'coauthorph', 'amazon-ratings', 'questions', 'chameleon-filtered',
                             'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'penn94',
                             'blogcatalog', 'flickr'], help='dataset')
parser.add_argument('--gsl', type=str, default='grcn', choices=['gt', 'prognn', 'gen', 'gaug', 'idgl', 'grcn', 'slaps',  'nodeformer', 'segsl', 'sublime', 'stable', 'cogsl'], help="Select methods")
parser.add_argument('--gnn', type=str, default='gcn', choices=['gcn', 'sgc', 'jknet', 'appnp', 'gprgnn', 'lpa', 'link'])
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import opengsl

conf = opengsl.config.load_conf(method=args.gnn, dataset=args.data)
# specify some settings
conf.analysis['load_graph'] = True
conf.analysis['load_graph_path'] = 'results/graph/{}'.format(args.gsl)
if args.gsl in ['sublime', 'idgl']:
    conf.dataset['normalize'] = False
else:
    conf.dataset['normalize'] = True
if args.gsl in ['grcn', 'sublime', 'idgl']:
    conf.dataset['add_loop'] = False
else:
    conf.dataset['add_loop'] = True
print(conf)
dataset = opengsl.data.Dataset(args.data, feat_norm=conf.dataset['feat_norm'], path='data')
method = eval('opengsl.method.{}(conf, dataset)'.format(args.gnn))
exp = opengsl.ExpManager(method,  save_path='records')
exp.run(n_runs=1, debug=args.debug)