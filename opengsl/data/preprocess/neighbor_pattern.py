import torch
import pickle
from datetime import datetime
import os.path as osp
import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid
from opengsl.utils import get_edge_homophily

D_cora = np.array([[0,0.5,0,0,0,0,0.5],[0.5,0,0.5,0,0,0,0],[0,0.5,0,0.5,0,0,0],[0,0,0.5,0,0.5,0,0],[0,0,0,0.5,0,0.5,0],[0,0,0,0,0.5,0,0.5],[0.5,0,0,0,0,0.5,0]])
D_citeseer = np.array([])


def add_edges_np(adj, labels, D, n_add_edges=None, target_homophily=None, seed=42):
    assert not (n_add_edges and target_homophily), "you cannot specify both"
    assert n_add_edges or target_homophily, "you must specify one"

    C = labels.max() + 1
    classes = [i for i in range(C)]
    n = len(labels)
    np.random.seed(seed)
    n_edges = adj.indices().shape[1]   # directed
    adj = adj.to_dense()

    # compute original homophily
    original_homophily = get_edge_homophily(labels, adj)
    if target_homophily:
        # compute n_add_edges
        n_add_edges = int((n_edges * original_homophily) / target_homophily) - n_edges
    else:
        # compute final homophily
        target_homophily = (n_edges * original_homophily) / (n_edges + n_add_edges)

    labels = labels.numpy()
    for k in range(n_add_edges):
        i = np.random.randint(0, n)
        yi = labels[i]
        yj = np.random.choice(classes, p=D[yi])
        while True:
            j = np.random.choice(np.where(labels == yj)[0])
            if adj[i, j] == 0:
                adj[i, j] = 1
                break
    return adj.to_sparse(), n_add_edges, target_homophily


if __name__ == '__main__':
    from opengsl.data.dataset import Dataset
    dataset = Dataset('cora', path='./')
    new_adj, homo = add_edges_np(dataset.adj.coalesce(), dataset.labels.cpu(), D_cora, n_add_edges=1003)
    print(new_adj, homo)
    print(get_edge_homophily(dataset.labels.cpu(), new_adj.to_dense()))
