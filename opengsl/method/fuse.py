def replace(adjs, sparse=False):
    return adjs['new']


def interpolation(adjs, sparse=False, lamb1=None, lamb2=None):
    return lamb1 * adjs['ori'] + lamb2 * adjs['new']


def multiply(adjs, sparse=False):
    return adjs['ori'] * adjs['new']