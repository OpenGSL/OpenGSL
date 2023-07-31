import ruamel.yaml as yaml
import argparse
import os


def load_conf(path:str = None, method:str = None, dataset:str = None):
    '''
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    '''
    if path == None and method == None:
        raise KeyError
    if path == None and dataset == None:
        raise KeyError
    if path == None:
        method_name = ['gcn', 'sgc', 'gat', 'jknet', 'appnp', 'gprgnn', 'prognn', 'idgl', 'grcn', 'gaug', 'slaps',
                       'gen', 'gt', 'nodeformer', 'cogsl', 'sublime', 'stable', 'segsl', 'lpa', 'link', 'wsgnn']
        data_name = ['cora', 'pubmed', 'citeseer','blogcatalog', 'flickr', 'amazon-ratings', 'questions', 'minesweeper', 'roman-empire', 'wiki-cooc']

        assert method in method_name
        assert dataset in data_name
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        if method in ["link", "lpa"]:
            path = os.path.join(dir, method, method+".yaml")
        else:
            path = os.path.join(dir, method, method+'_'+dataset+".yaml")

        if os.path.exists(path) == False:
            raise KeyError("The configuration file is not provided.")
    
    conf = open(path, "r").read()
    conf = yaml.safe_load(conf)
    conf = argparse.Namespace(**conf)

    return conf


def save_conf(path, conf):
    '''
    Function to save the config file.

    Parameters
    ----------
    path : str
        Path to save config file.
    conf : dict
        The config dict.

    '''
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(vars(conf), f)