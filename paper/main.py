import os
import argparse
import ruamel.yaml as yaml


def main(args):
    from pipeline.ExpManager import ExpManager

    print(args)

    # load config
    if args.config != '':
        conf = open(args.config, "r").read()
        conf = yaml.safe_load(conf)
        conf = argparse.Namespace(**conf)
        print(conf)
    else:
        conf = None

    a = ExpManager(conf, method=args.method, data=args.data, n_splits=args.n_splits, n_runs=args.n_runs, save=args.save,
                   debug=args.debug, verbose=args.verbose)
    if 'sweep' in conf.analysis and conf.analysis['sweep']:
        a.sweep()
    else:
        a.run()



if __name__ == '__main__':
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
    parser.add_argument('--config', type=str, default='configs/gcn/gcn_template.yaml', help="Config file used for specific model training.")
    parser.add_argument('--n_runs', type=int, default=1,
                        help="number of exps per data split")
    parser.add_argument('--n_splits', type=int, default=1,
                        help="number of different data splits (For citation datasets you get the same split "
                             "unless you have re_split=true in the config file)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args)
