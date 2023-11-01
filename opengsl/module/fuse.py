class Replace:
    '''
    This fuse object uses the new structure directly.
    '''

    def __init__(self):
        pass

    def __call__(self, adj, raw_adj):
        '''
        Parameters
        ----------
        adj : torch.tensor
            New structure.
        raw_adj : torch.tensor
            Original structure.

        Returns
        -------
        fused_adj : torch.tensor
            Equal to `adj`.
        '''
        fused_adj = adj
        return fused_adj


class Interpolate:
    '''
    Linear Interpolation for the new structure and original structure.

    Parameters
    ----------
    lamb1 : float
        The weight for the new structure.
    lamb2 : float
        The weight for the original structure.
    '''
    def __init__(self, lamb1, lamb2=None):
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        if lamb2 is None:
            self.lamb2 = 1 - self.lamb1

    def __call__(self, adj, raw_adj):
        '''
        Parameters
        ----------
        adj : torch.tensor
            New structure.
        raw_adj : torch.tensor
            Original structure.

        Returns
        -------
        fused_adj : torch.tensor
            Fused adj.
        '''
        fused_adj = self.lamb1 * adj + self.lamb2 * raw_adj
        return fused_adj


class Multiply:
    '''
    Multiplication for the new structure and original structure.
    '''

    def __init__(self):
        pass

    def __call__(self, adj, raw_adj):
        '''
        Parameters
        ----------
        adj : torch.tensor
            New structure.
        raw_adj : torch.tensor
            Original structure.

        Returns
        -------
        fused_adj : torch.tensor
            Fused adj.
        '''
        return adj * raw_adj