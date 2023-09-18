'''
load data via pyg
'''

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor, AttributedGraphDataset
from .csbm_load import dataset_ContextualSBM
import os


def pyg_load_dataset(name, path='./data/'):
    dic = {'cora': 'Cora',
           'citeseer': 'CiteSeer',
           'pubmed': 'PubMed',
           'amazoncom': 'Computers',
           'amazonpho': 'Photo',
           'coauthorcs': 'CS',
           'coauthorph': 'Physics',
           'wikics': 'WikiCS',
           'chameleon': 'Chameleon',
           'squirrel': 'Squirrel',
           'cornell': 'Cornell',
           'texas': 'Texas',
           'wisconsin': 'Wisconsin',
           'actor': 'Actor',
           'blogcatalog':'blogcatalog',
           'flickr':'flickr'}
    if 'csbm' in name:
        name = name
    else:
        name = dic[name]

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=os.path.join(path, name), name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=os.path.join(path, name), name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=os.path.join(path, name), name=name)
    elif name in ['WikiCS']:
        dataset = WikiCS(root=os.path.join(path, name))
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        dataset = WikipediaNetwork(root=os.path.join(path, name), name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=os.path.join(path, name), name=name)
    elif name == 'Actor':
        dataset = Actor(root=os.path.join(path, name))
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=os.path.join(path, name), name=name)
    elif 'csbm' in name:
        dataset = dataset_ContextualSBM(root=path, name=name)
    else:
        exit("wrong dataset")
    return dataset