import torch
from opengsl.module.functional import normalize, symmetry, knn, enn, apply_non_linearity
from opengsl.module.metric import Cosine
import torch.nn as nn

class Normalize(nn.Module):
    '''
    Normalize the feature matrix or adj matrix.

    Parameters
    ----------
    style: str
        If set as `row`, matrix will be row-wise normalized.
        If set as `symmetric`, matrix will be normalized as in GCN.
        If set as `softmax`, matrix will be normalized using softmax.
        If set as `row-norm`, matrix will be normalized using `F.normalize` in pytorch.
    add_loop : bool
        Whether to add self loop.
    p : float
        The exponent value in the norm formulation. Onlu used when style is set as `row-norm`.
    '''

    def __init__(self, style='symmetric', add_loop=True, p=None):
        super(Normalize, self).__init__()
        self.style = style
        self.add_loop = add_loop
        self.p = p

    def forward(self, adj):
        '''
        Parameters
        ----------
        adj : torch.tensor
            Feature matrix or adj matrix to normalize. Note that either sparse or dense form is supported.

        Returns
        -------
        normalized_mx : torch.tensor
            The normalized matrix.
        '''
        return normalize(adj, self.style, self.add_loop, self.p)


class Symmetry(nn.Module):
    '''
    Symmetry the feature matrix or adj matrix.

    Parameters
    ----------
    i: int
        The denominator after summation. 
    '''

    def __init__(self, i=2):
        super(Symmetry, self).__init__()
        self.i = i

    def forward(self, adj):
        '''
        Parameters
        ----------
        adj : torch.tensor
            Feature matrix or adj matrix to symmetry. Note that either sparse or dense form is supported.

        Returns
        -------
        normalized_mx : torch.tensor
            The symmetric matrix.
        '''
        return symmetry(adj, self.i)


class KNN(nn.Module):
    '''
    Select KNN matrix each row.

    Parameters
    ----------
    K : int
        Number of neighbors.
    self_loop : bool
        Whether to include self loops.
    set_value : float
        Specify the value for selected elements. The original value will be used if set to `None`.
    metric : str
        The similarity function.
    sparse_out : bool
        Whether to return adj in sparse form.
    '''

    def __init__(self, K, self_loop=True, set_value=None, metric='cosine', sparse_out=False):
        super(KNN, self).__init__()
        self.K = K
        self.self_loop = self_loop
        self.set_value = set_value
        self.sparse_out = sparse_out
        if metric:
            if metric == 'cosine':
                self.metric = Cosine()

    def forward(self, x=None, adj=None):
        '''
        Generate KNN matrix given node embeddings or adj matrix. Pairwise similarities will first 
        be calculated if ``x`` is given.
        Parameters
        ----------
        x : torch.tensor
            Node embeddings.
        adj : torch.tensor
            Input adj. Note only dense form is supported currently.
    
        Returns
        -------
        knn_adj : torch.tensor
            KNN matrix.
        '''
        assert not (x is None and adj is None)
        if x is not None:
            dist = self.metric(x)
        else:
            dist = adj
        return knn(dist, self.K, self.self_loop, set_value=self.set_value, sparse_out=self.sparse_out)


class EpsilonNN(nn.Module):
    '''
    Select elements according to the threshold.

    Parameters
    ----------
    epsilon : float
        Threshold.
    set_value : float
        Specify the value for selected elements. The original value will be used if set to `None`.
    '''

    def __init__(self, epsilon, set_value=None):
        super(EpsilonNN, self).__init__()
        self.epsilon = epsilon
        self.set_value = set_value

    def forward(self, adj):
        '''
        Generate matrix given adj matrix. 
        Parameters
        ----------
        adj : torch.tensor
            Input adj. Note that either sparse or dense form is supported.

        Returns
        -------
        new_adj : torch.tensor
            Adj with elements larger than the threshold.
        '''
        return enn(adj, self.epsilon, self.set_value)


class NonLinear(nn.Module):
    '''
    Nonlinear function.

    Parameters
    ----------
    non_linearity : str
        Specify the function.
    i : int
        Integer used in elu.
    '''

    def __init__(self, non_linearity, i=None):
        super(NonLinear, self).__init__()
        self.non_linearity = non_linearity
        self.i = i

    def forward(self, adj):
        '''
        Apply the nonlinear function. 
        Parameters
        ----------
        adj : torch.tensor
            Input adj. Note that either sparse or dense form is supported.

        Returns
        -------
        new_adj : torch.tensor
        '''
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