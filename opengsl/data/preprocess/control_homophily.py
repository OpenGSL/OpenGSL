from opengsl.utils.utils import get_homophily
import numpy as np


def control_homophily(adj, labels, homophily):
    '''
    Control the homophily of original structure by adding edges.
    More ways to add perturbations will be implemented soon.

    Parameters
    ----------
    adj : torch.tensor
        The original structure in sparse form.
    labels : torch.tensor
        Ground truth labels.
    homophily : float
        Homophily ratio.

    Returns
    -------
    new_adj : torch.tensor
        The perturbed structure in sparse form.

    '''
    np.random.seed(0)
    # change homophily through adding edges
    adj = adj.to_dense()
    n_edges = adj.sum()/2
    n_nodes = len(labels)
    homophily_orig = get_homophily(labels, adj, 'edge')
    # print(homophily_orig)
    if homophily<homophily_orig:
        # add noisy edges
        n_add_edges = int(n_edges*homophily_orig/homophily-n_edges)
        while n_add_edges>0:
            u = np.random.randint(0, n_nodes)
            vs = np.arange(0, n_nodes)[labels!=labels[u]]
            v = np.random.choice(vs)
            if adj[u, v]==0:
                adj[u,v]=adj[v,u]=1
                n_add_edges-=1
    if homophily>homophily_orig:
        # add helpful edges
        n_add_edges = int(n_edges*(1-homophily_orig)/(1-homophily)-n_edges)
        while n_add_edges > 0:
            u = np.random.randint(0, n_nodes)
            vs = np.arange(0, n_nodes)[labels==labels[u]]
            v = np.random.choice(vs)
            if u==v:
                continue
            if adj[u,v]==0:
                adj[u,v]=adj[v,u]=1
                n_add_edges -= 1
    return adj.to_sparse()



