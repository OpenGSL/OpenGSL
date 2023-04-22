import numpy as np
import os
import torch
import dgl

def hetero_load(name, path='./data/hetero_data'):
    data = np.load(os.path.join(path, f'{name.replace("-", "_")}.npz'))
    node_features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges'])
    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)
    graph = dgl.to_bidirected(graph)


    num_classes = len(labels.unique())
    num_targets = 1 if num_classes == 2 else num_classes
    if num_targets == 1:
        labels = labels.float()

    train_masks = torch.tensor(data['train_masks']).transpose(0,1)
    val_masks = torch.tensor(data['val_masks']).transpose(0,1)
    test_masks = torch.tensor(data['test_masks']).transpose(0,1)

    graph.ndata['label'] = labels
    graph.ndata['feat'] = node_features
    graph.ndata['train_mask'] = train_masks
    graph.ndata['val_mask'] = val_masks
    graph.ndata['test_mask'] = test_masks
    return graph