import os
import importlib
import argparse
import ruamel.yaml as yaml
from data.dataset import Dataset
from utils.utils import set_seed

def main(args):

    # load config
    if args.config != '':
        conf = open(args.config, "r").read()
        conf = yaml.safe_load(conf)
        conf = argparse.Namespace(**conf)
        print(conf)
    else:
        conf = None

    # load data
    dataset = Dataset(args.data, feat_norm=conf.dataset['feat_norm'], verbose=args.verbose, n_splits=args.n_splits, cora_split=conf.dataset['cora_split'])

    set_seed(0)
    from solvers.solver_grcn1 import GRCNSolver
    a=GRCNSolver(conf,dataset)
    print(a.learn(debug=args.debug))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora',
                        choices=['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho',
                                 'coauthorcs', 'coauthorph', 'amazon-ratings', 'questions', 'chameleon-filtered',
                                 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'penn94'], help='dataset')
    parser.add_argument('--solver', type=str, default='gcn',
                        choices=['gcn', 'appnp', 'gt', 'gat', 'prognn', 'gen', 'gaug', 'idgl', 'grcn', 'sgc'], help="The version of solver")
    parser.add_argument('--config', type=str, default='configs/gcn/gcn_template.yaml', help="Config file used for specific model training.")
    parser.add_argument('--n_runs', type=int, default=1,
                        help="number of exps per data split")
    parser.add_argument('--n_splits', type=int, default=1,
                        help="number of different data splits (For citation datasets you get the same split "
                             "unless you have re_split=true in the config file)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_graph', action='store_true')
    parser.add_argument('--not_norm_feats', action='store_true', help='whether to normalize the feature matrix')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["WANDB_API_KEY"] = 'b87288e13764b564a70c64817827e73228ae48ec'
    os.environ["WANDB_MODE"] = "offline"

    main(args)
