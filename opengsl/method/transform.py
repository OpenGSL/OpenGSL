import torch
from opengsl.method.functional import normalize, symmetry, knn, enn, apply_non_linearity
from opengsl.method.metric import Cosine


class Normalize:

    def __init__(self, style='symmetric', add_loop=True, p=None):
        self.style = style
        self.add_loop = add_loop
        self.p = p

    def __call__(self, adj):
        return normalize(adj, self.style, self.add_loop, self.p)


class Symmetry:

    def __init__(self, i=2):
        self.i = i

    def __call__(self, adj):
        return symmetry(adj, self.i)


class KNN:

    def __init__(self, K, self_loop=True, set_value=None, metric='cosine', sparse_out=False):
        self.K = K
        self.self_loop = self_loop
        self.set_value = set_value
        self.sparse_out = sparse_out
        if metric:
            if metric == 'cosine':
                self.metric = Cosine()

    def __call__(self, x=None, adj=None):
        assert not (x is None and adj is None)
        if x is not None:
            dist = self.metric(x)
            if adj:
                # TODO add new edge on raw adj
                pass
        else:
            dist = adj
        return knn(dist, self.K, self.self_loop, set_value=self.set_value, sparse_out=self.sparse_out)


class EpsilonNN:

    def __init__(self, epsilon, set_value=None):
        self.epsilon = epsilon
        self.set_value = set_value

    def __call__(self, adj):
        return enn(adj, self.epsilon, self.set_value)


class NonLinear:

    def __init__(self, non_linearity, i=None):
        self.non_linearity = non_linearity
        self.i = i

    def __call__(self, adj):
        return apply_non_linearity(adj, self.non_linearity, self.i)


if __name__ == '__main__':
    from torch_geometric import seed_everything
    seed_everything(42)
    # adj = torch.rand(5, 5).to_sparse()
    # adj = torch.sparse.FloatTensor(torch.tensor([[0,0,1,1,2,2,3,3,4],[1,2,3,4,0,1,2,3,3]]), torch.tensor([1,1,1,1,1,1,1,1,1]), [5,5])
    adj = torch.rand(3, 3)
    print(adj)
    f = Symmetry()
    print(f(adj))