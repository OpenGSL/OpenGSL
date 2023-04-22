'''
Here we provide another way to load data with pyg. This is needed when running grcn because grcn uses the unnormalized
feature, which is not supported in dgl.
'''

from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor


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
           'actor': 'Actor'}
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
    else:
        exit("wrong dataset")
    return dataset