import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import torch


def knn(feature, k):
    # Generate a knn graph for input feature matrix. Note that the graph does not contain self loop.
    device = feature.device
    n_nodes = feature.shape[0]
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    dist = cos(feature.detach().cpu().numpy())
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(n_nodes).repeat(k + 1), col] = 1
    # remove self-loop
    np.fill_diagonal(adj, 0)
    adj = torch.tensor(adj).to(device)
    return adj

if __name__ == '__main__':
    a = torch.rand(5,3)
    print(knn(a,2))