import deeprobust.graph.utils
import torch
from opengsl.utils.utils import scipy_sparse_to_sparse_tensor, sparse_tensor_to_scipy_sparse
from deeprobust.graph.global_attack import MetaApprox, Metattack, Random
from deeprobust.graph.utils import preprocess
import numpy as np
from deeprobust.graph.defense import GCN


def metattack(adj, features, labels, train_mask, val_mask, test_mask, ptb_rate):

    device = adj.device
    adj = adj.to_dense().cpu()
    features = features.cpu()
    labels = labels.cpu()
    nclass = labels.max()+1
    idx_unlabeled = np.union1d(val_mask, test_mask)
    perturbations = int(ptb_rate * (adj.sum() // 2))

    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0.5, with_relu=True, with_bias=True, device=device, weight_decay=5e-4)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, train_mask)

    # Setup Attack Model
    model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                      attack_structure=True, attack_features=False, device=device, lambda_=0)
    model = model.to(device)

    # Attack
    model.attack(features, adj, labels, train_mask, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj

    return modified_adj


def random_attack(adj, conf, device):
    model = Random()

    n_perturbations = int(conf.attack['ptb_rate'] * (adj.sum() // 2))
    model.attack(sparse_tensor_to_scipy_sparse(adj), n_perturbations)
    return scipy_sparse_to_sparse_tensor(model.modified_adj).to_dense().to(device)

if __name__ == '__main__':
    import numpy as np
    from deeprobust.graph.data import Dataset
    from deeprobust.graph.defense import GCN
    from deeprobust.graph.global_attack import Metattack
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj, features, labels = preprocess(adj, features, labels)

     # Setup Surrogate model
    device = torch.device('cuda:0')
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                         nhid=16, dropout=0.5, with_relu=True, with_bias=True, device=device, weight_decay=5e-4)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
     # Setup Attack Model
    model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                           attack_structure=True, attack_features=False, device=device, lambda_=0)
    model = model.to(device)
     # Attack
    ptb_rate = 0.2
    perturbations = int(ptb_rate * (adj.sum() // 2))
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj
    print(modified_adj)