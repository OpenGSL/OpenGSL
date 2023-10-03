import torch
from opengsl.method.functional import normalize, symmetry, knn, apply_non_linearity


class Normalize:

    def __init__(self, style='symmetric', add_loop=True, p=None):
        self.style = style
        self.add_loop = add_loop
        self.p = p

    def __call__(self, adj):
        return normalize(adj, self.style, self.add_loop, self.p)


class Symmetry:

    def __init__(self):
        pass

    def __call__(self, adj):
        return symmetry(adj)


class KNN:

    def __init__(self, K):
        self.K = K

    def __call__(self, adj):
        return knn(adj, self.K)


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