import torch
from dgl.transforms import AddReverse, ToSimple        # dgl0.9

def load_ogb(name):
    """
    name in [ ogbn-arxiv, ogbn-products ]
    """
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)

    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    
    #num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    num_labels = data.num_classes

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['label'] = labels.squeeze()

    trans1 = AddReverse()
    trans2 = ToSimple()
    graph = trans2(trans1(graph))  # dgl0.9
    
    print('finish constructing', name)

    return data, graph