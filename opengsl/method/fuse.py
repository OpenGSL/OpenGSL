class Replace:

    def __init__(self):
        pass

    def __call__(self, adj, raw_adj):
        return adj


class Interpolate:
    def __init__(self, lamb1, lamb2=None):
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        if lamb2 is None:
            self.lamb2 = 1 - self.lamb1

    def __call__(self, adj, raw_adj):
        return self.lamb1 * adj + self.lamb2 * raw_adj


class Multiply:

    def __init__(self):
        pass

    def __call__(self, adj, raw_adj):
        return adj * raw_adj