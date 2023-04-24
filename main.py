import os
import importlib
import argparse
import ruamel.yaml as yaml

def main(args):

    print(args)
    a = importlib.import_module('.solver_'+args.solver, package='solvers')

    if args.config != '':
        conf = open(args.config, "r").read()
        conf = yaml.safe_load(conf)
        conf = argparse.Namespace(**conf)
        print(conf)
    else:
        conf = None

    solver = a.Solver(args, conf)
    solver.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora',
                        choices=['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho',
                                 'coauthorcs', 'coauthorph', 'amazon-ratings', 'questions', 'chameleon-filtered',
                                 'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc'], help='dataset')
    parser.add_argument('--solver', type=str, default='gcn',
                        choices=['gcn', 'appnp', 'gt', 'gat', 'prognn', 'gen', 'gaug', 'idgl', 'grcn'], help="The version of solver")
    parser.add_argument('--config', type=str, default='configs/gcn/gcn_template.yaml', help="Config file used for specific model training.")
    parser.add_argument('--n_runs', type=int, default=1,
                        help="number of exps per data split")
    parser.add_argument('--n_splits', type=int, default=1,
                        help="number of different data splits (For citation datasets you get the same split "
                             "unless you have re_split=true in the config file)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--not_norm_feats', action='store_true', help='whether to normalize the feature matrix')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
    args = parser.parse_args()

    main(args)
