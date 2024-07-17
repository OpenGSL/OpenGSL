import argparse
import os
import sys
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='citeseer', help='dataset')
parser.add_argument('--method', type=str, default='prose', help="Select methods")
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--debug', action='store_false')
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
parser.add_argument('--n_splits', type=int, default=1)
parser.add_argument('--n_runs', type=int, default=5)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from opengsl.config import load_conf
from opengsl.data import Dataset
from opengsl import ExpManager
from opengsl.module import *

if args.config is None:
    conf = load_conf(method=args.method, dataset=args.data)
else:
    conf = load_conf(args.config)
# conf.analysis['save_graph'] = False
print(conf)

if 'without_structure' in conf.dataset and conf.dataset['without_structure']:
    without_structure = conf.dataset['without_structure']
else:
    without_structure = None
dataset = Dataset(args.data, path='data', n_splits=args.n_splits, **conf.dataset)


method = eval('{}Solver(conf, dataset)'.format(args.method.upper()))
exp = ExpManager(method)
acc_save, std_save = exp.run(n_runs=args.n_runs, debug=args.debug)
text = '{:.2f} ± {:.2f}'.format(acc_save, std_save)

if not os.path.exists('results'):
    os.makedirs('results')
if os.path.exists('results/performance.csv'):
    records = pd.read_csv('results/performance.csv')
    records.loc[len(records)] = {'method':args.method, 'data':args.data, 'acc':text}
    records.to_csv('results/performance.csv', index=False)
else:
    records = pd.DataFrame([[args.method, args.data, text]], columns=['method', 'data', 'acc'])
    records.to_csv('results/performance.csv', index=False)