import ruamel.yaml as yaml
import argparse
import os

def load_conf(path:str = None, method:str = None, dataset:str = None):
    if path == None and method == None:
        raise KeyError
    if path == None:
        method_name = ['gcn', 'sgc', 'gat', 'jknet', 'appnp', 'gprgnn', 'prognn', 'idgl', 'grcn', 'gaug', 'slaps', 'gen', 'gt', 'nodeformer', 'cogsl', 'sublime', 'stable', 'segsl']
        data_name = ['cora', 'pubmed', 'citeseer','blogcatalog', 'flickr', 'amazon-ratings', 'questions', 'minesweeper', 'roman-empire', 'wiki-cooc']

        assert method in method_name
        assert dataset in data_name
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        if method in ["link", "lpa"]:
            path = os.path.join(dir, method, method+".yaml")
        else:
            path = os.path.join(dir, method, method+'_'+dataset+".yaml")


    conf = open(path, "r").read()
    conf = yaml.safe_load(conf)
    conf = argparse.Namespace(**conf)

    return conf