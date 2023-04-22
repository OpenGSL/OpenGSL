import torch
import dgl
import dgl.data
import os
from .load_cora import CoraData
from .ogb_data import load_ogb


def load_dataset(name='cora', reload_gs=False, graph_fn=""):

    dic = {'cora':'CoraGraphDataset',
           'citeseer':'CiteseerGraphDataset',
           'pubmed':'PubmedGraphDataset',
           'amazoncom':'AmazonCoBuyComputerDataset',
           'amazonpho':'AmazonCoBuyPhotoDataset',
           'coauthorcs':'CoauthorCSDataset',
           'coauthorph':'CoauthorPhysicsDataset',
           'wikics':'WikiCSDataset'}

    if name == 'raw_cora':
        dataset = CoraData('data/cora', True)
        g = dataset[0]
    # elif name == 'cora':
    #     dataset = dgl.data.CoraGraphDataset()
    #     g = dataset[0]
    # elif name == 'citeseer':
    #     dataset = dgl.data.CiteseerGraphDataset()
    #     g = dataset[0]
    # elif name == 'pubmed':
    #     dataset = dgl.data.PubmedGraphDataset()
    #     g = dataset[0]
    # elif name == 'amazoncom':
    #     dataset = dgl.data.AmazonCoBuyComputerDataset()
    #     g = dataset[0]
    # elif name == 'amazonpho':
    #     dataset = dgl.data.AmazonCoBuyPhotoDataset()
    #     g = dataset[0]
    # elif name == 'coauthorcs':
    #     dataset = dgl.data.CoauthorCSDataset()
    #     g = dataset[0]
    # elif name == 'coauthorph':
    #     dataset = dgl.data.CoauthorPhysicsDataset()
    #     g = dataset[0]
    # elif name == 'wikics':
    #     dataset = dgl.data.WikiCSDataset()
    #     g = dataset[0]
    elif name == "ogbn-arxiv":
        dataset, g = load_ogb(name='ogbn-arxiv')
    elif name in dic.keys():
        dataset = eval('dgl.data.'+dic[name]+'()')
        g = dataset[0]
    else:
        raise NotImplementedError("Dataset named {} is not available.".format(name))

    '''
    if reload_gs:
        graph_fn = graph_fn.strip()
        assert os.path.exists(graph_fn)
    
        if graph_fn.endswith('dgl'):
            raise NotImplementedError("Have not implemented dgl graph load and save")
        else:
            edge_d = torch.load(graph_fn)
            n_edges = g.number_of_edges()
            g.remove_edges(list(range(n_edges)))
            if isinstance(edge_d, dict):
                g.add_edges(u=edge_d['u'].long(), v=edge_d['v'].long())
            elif torch.is_tensor(edge_d):
                edge_d = edge_d.to_sparse()
                indices = edge_d.indices()
                g.add_edges(u=indices[0].long(), v=indices[1].long())
    '''
    
    return dataset, g