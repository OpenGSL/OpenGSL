import numpy as np
import os
import torch
import dgl

def hetero_load(name, path='./data/hetero_data'):
    data = np.load(os.path.join(path, f'{name.replace("-", "_")}.npz'))
    node_features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges'])
    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    train_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in train_masks]
    val_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in val_masks]
    test_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in test_masks]


    n_nodes = node_features.shape[0]
    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)
    graph = dgl.to_bidirected(graph)
    adj = graph.adj()

    num_classes = len(labels.unique())
    num_targets = 1 if num_classes == 2 else num_classes
    if num_targets == 1:
        labels = labels.float()


    return node_features, adj, labels, (train_indices, val_indices, test_indices)