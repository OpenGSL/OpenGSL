'''
load data via pyg
'''

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor, AttributedGraphDataset, TUDataset
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
    if name in dic.keys():
        name = dic[name]
    else:
        name = name

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=path, name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=path, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=path, name=name)
    elif name in ['WikiCS']:
        dataset = WikiCS(root=os.path.join(path, name))
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=path, name=name)
    elif name == 'Actor':
        dataset = Actor(root=os.path.join(path, name))
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=path, name=name)
    elif 'csbm' in name:
        dataset = dataset_ContextualSBM(root=path, name=name)
    elif name in ["IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB", "DBLP_v1", "DD", "ENZYMES", "PROTEINS", "MUTAG", "NCI1", "NCI109", "Mutagenicity", "FRANKENSTEIN"]:
        dataset = TUDataset(root=path, name=name, use_edge_attr=False)
    return dataset