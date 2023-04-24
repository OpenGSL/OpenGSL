import dgl.random
import torch
import os
import numpy as np
import scipy.sparse as sp
import random


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


# def accuracy(labels, logits):
#     '''
#     :param logits: torch.tensor logits from the model (N,nclass)
#     :param labels: torch.tensor ground truth labels (N,)
#     :return: accuracy
#     '''
#     _, indices = torch.max(logits, dim=1)
#     correct = torch.sum(indices == labels)
#     return correct.item() * 1.0 / len(labels)

def accuracy(labels, logits):
    return np.sum(logits.argmax(1)==labels)/len(labels)

def inner_list_2_tensor(nested_list):
    for i in range(len(nested_list)):
        nested_list[i] = torch.Tensor(nested_list[i]).long()
    return nested_list

def edge_list_2_adj_list(edge_list, n_nodes):
        u, v = edge_list
        adj_list = [[] for _ in range(n_nodes)]
        for i in range(u.shape[0]):
            adj_list[u[i]].append(v[i])
        adj_list = inner_list_2_tensor(adj_list)
        return adj_list

def adj_list_2_edge_list(adj_list):
    u_, v_ = [], []
    for u in range(len(adj_list)):
        for v in adj_list[u]:
            u_.append(u)
            v_.append(v)
    u_ = torch.Tensor(u_).long()
    v_ = torch.Tensor(v_).long()
    return (u_, v_)
    
def count_edge(adj_list):
    return sum([len(neighs) for neighs in adj_list])
    
def normalize(adj, add_loop=True):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0]).to(device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A

def normalize_sp_tensor(adj, add_loop=True):
    device = adj.device
    adj = sparse_tensor_to_scipy_sparse(adj)
    adj = sparse_normalize(adj, add_loop)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    return adj

def sparse_normalize(adj, add_loop=True):
    mx = adj + sp.eye(adj.shape[0]) if add_loop else adj
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    new = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return new

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_tensor_to_scipy_sparse(adj):
    adj = adj.cpu()
    row = adj.coalesce().indices()[0].numpy()
    col = adj.coalesce().indices()[1].numpy()
    values = adj.coalesce().values().numpy()
    return sp.coo_matrix((values, (row, col)), shape=adj.shape)


def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
    """ schedule the learning rate with the sigmoid function.
    The learning rate will start with near zero and end with near lr """
    factors = torch.FloatTensor(np.arange(n_epochs))
    factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
    factors = torch.sigmoid(factors)
    # range the factors to [0, 1]
    factors = (factors - factors[0]) / (factors[-1] - factors[0])
    lr_schedule = factors * lr
    return lr_schedule


def normalize_feat(mx):
    """Row-normalize sparse matrix.
    """
    r_sum = np.array(mx.sum(1))
    r_inv = np.power(r_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_feats(x):
    rowsum = x.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    X = r_mat_inv @ x
    return X

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_node_homophily(label, adj):
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = (np.multiply((label == label.T), adj)).sum(axis=1)
    d = adj.sum(axis=1)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1. / d[i])
    return np.mean(homos)

def get_dict(adj):
    g = {}
    for i in range(adj.shape[0]):
        g[i] = []
    mn = adj.triu(1).nonzero().tolist()  # 每个边只会出现一次，列表不用去重
    for m, n in mn:
        g[m].append(n)
        g[n].append(m)
    return g

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

def set_seed(seed):
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_edge_homophily(label, adj):
    num_edge = adj.sum()
    cnt = 0
    for i, j in adj.nonzero():
        if label[i] == label[j]:
            cnt += adj[i, j]
    return cnt/num_edge

def get_homophily(label, adj, type='node'):
    # np.fill_diagonal(adj, 0)
    return eval('get_'+type+'_homophily(label, adj)')