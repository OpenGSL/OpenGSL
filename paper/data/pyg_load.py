'''
load data via pyg
'''

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor, AttributedGraphDataset


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
    name = dic[name]

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=path+name, name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=path+name, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=path+name, name=name)
    elif name in ['WikiCS']:
        dataset = WikiCS(root=path+name)
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        dataset = WikipediaNetwork(root=path+name, name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=path+name, name=name)
    elif name == 'Actor':
        dataset = Actor(root=path+name)
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=path+name, name=name)
    else:
        exit("wrong dataset")
    return dataset