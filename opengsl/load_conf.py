import ruamel.yaml as yaml
import argparse
import os

def load_conf(path:str = None, method:str = None, dataset:str = None):
    if path == None and method == None:
        raise KeyError
    if path == None:
        method_name = ['gcn', 'sgc', 'gat', 'jknet', 'appnp', 'gprgnn', 'prognn', 'idgl', 'grcn', 'gaug', 'slaps', 'gen', 'gt', 'nodeformer', 'cogsl', 'sublime', 'stable', 'segsl']
        data_name = ['cora', 'pubmed', 'citeseer','blogcatalog', 'flickr', 'amazon-ratings', 'questions', 'minesweeper', 'roman-empire', 'wiki-cooc']
        data_cora = ['cora', 'pubmed', 'citeseer']
        data_blog = ['blogcatalog', 'flickr']
        data_hetero = ['amazon-ratings', 'questions', 'minesweeper', 'roman-empire', 'wiki-cooc']
        assert method in method_name
        assert dataset in data_name
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        dir = os.path.join(dir, method)
        if method == "gcn":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "sgc":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "gat":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "jknet":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "appnp":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "gprgnn":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "prognn":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "idgl":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "grcn":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "gaug":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "slaps":
            if dataset in data_blog:
                path = os.path.join(dir, method+"_blog.yaml")
            elif dataset in data_hetero:
                path = os.path.join(dir, method+"_wiki.yaml")
            else:
                path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "gen":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "gt":
            if dataset in data_cora:
                path = os.path.join(dir, method+"_cora.yaml")
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "nodeformer":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "cogsl":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "sublime":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "stable":
            path = os.path.join(dir, method+'_'+dataset+".yaml")
        elif method == "segsl":
            path = os.path.join(dir, method+'_'+dataset+".yaml")


    conf = open(path, "r").read()
    conf = yaml.safe_load(conf)
    conf = argparse.Namespace(**conf)

    return conf