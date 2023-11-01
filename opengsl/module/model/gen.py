import numpy as np
from collections import Counter
import torch
from opengsl.module.transform import EpsilonNN


class EstimateAdj:
    def __init__(self, n_classes, adj, train_mask, labels, homophily):
        self.num_class = n_classes
        self.num_node = adj.shape[0]
        self.idx_train = train_mask
        self.label = labels.cpu().numpy()
        self.adj = adj.cpu().numpy()

        self.output = None
        self.iterations = 0
        self.count = 0

        self.homophily = homophily

    def reset_obs(self):
        self.count = 0
        self.N = 0
        self.E = np.zeros((self.num_node, self.num_node), dtype=np.int64)

    def update_obs(self, graph):
        self.E += graph
        self.N += 1

    def revise_pred(self):
        # For the training node, GT is used, and for the unlabeled node, the predicted label is used
        self.output[self.idx_train] = self.label[self.idx_train]

    def E_step(self, Q):
        """Run the Expectation(E) step of the EM algorithm.
        Parameters
        ----------
        Q:
            The current estimation that each edge is actually present (numpy.array)

        Returns
        ----------
        alpha:
            The estimation of true-positive rate (float)
        beta：
            The estimation of false-positive rate (float)
        O:
            The estimation of network model parameters (numpy.array)
        """
        # Temporary variables to hold the numerators and denominators of alpha and beta
        an = Q * self.E
        an = np.triu(an, 1).sum()
        bn = (1 - Q) * self.E
        bn = np.triu(bn, 1).sum()
        ad = Q * self.N
        ad = np.triu(ad, 1).sum()
        bd = (1 - Q) * self.N
        bd = np.triu(bd, 1).sum()

        # Calculate alpha, beta
        alpha = an * 1. / (ad)
        beta = bn * 1. / (bd)

        O = np.zeros((self.num_class, self.num_class))

        n = []
        counter = Counter(self.output)
        for i in range(self.num_class):
            n.append(counter[i])

        a = self.output.repeat(self.num_node).reshape(self.num_node, -1)
        for j in range(self.num_class):
            c = (a == j)
            for i in range(j + 1):
                b = (a == i)
                O[i, j] = np.triu((b & c.T) * Q, 1).sum()
                if i == j:
                    O[j, j] = 2. / (n[j] * (n[j] - 1)) * O[j, j]
                else:
                    O[i, j] = 1. / (n[i] * n[j]) * O[i, j]
        return (alpha, beta, O)

    def M_step(self, alpha, beta, O):
        """Run the Maximization(M) step of the EM algorithm.
        """
        O += O.T - np.diag(O.diagonal())   #使有对角元素的上三角变为对称矩阵

        row = self.output.repeat(self.num_node)
        col = np.tile(self.output, self.num_node)
        tmp = O[row, col].reshape(self.num_node, -1)

        p1 = tmp * np.power(alpha, self.E) * np.power(1 - alpha, self.N - self.E)
        p2 = (1 - tmp) * np.power(beta, self.E) * np.power(1 - beta, self.N - self.E)
        Q = p1 * 1. / (p1 + p2 * 1.)
        return Q

    def EM(self, output, tolerance=.000001):
        """Run the complete EM algorithm.
        Parameters
        ----------
        tolerance:
            Determine the tolerance in the variantions of alpha, beta and O, which is acceptable to stop iterating (float)
        seed:
            seed for np.random.seed (int)

        Returns
        ----------
        iterations:
            The number of iterations to achieve the tolerance on the parameters (int)
        """
        # Record previous values to confirm convergence
        alpha_p = 0
        beta_p = 0

        self.output = output
        self.revise_pred()

        # Do an initial E-step with random alpha, beta and O
        # Beta must be smalller than alpha
        beta, alpha = np.sort(np.random.rand(2))
        O = np.triu(np.random.rand(self.num_class, self.num_class))   #有对角元素的上三角

        # Calculate initial Q
        Q = self.M_step(alpha, beta, O)

        while abs(alpha_p - alpha) > tolerance or abs(beta_p - beta) > tolerance:
            alpha_p = alpha
            beta_p = beta
            alpha, beta, O = self.E_step(Q)
            Q = self.M_step(alpha, beta, O)
            self.iterations += 1
            self.count += 1
            #print(self.iterations,alpha,beta)

        if self.homophily > 0.5:
            # Make sure that the initial edge will be preserved
            Q += self.adj
        return (alpha, beta, O, Q, self.iterations)


def prob_to_adj(mx, threshold):
    mx = np.triu(mx, 1)   # the 2 steps here del the self loop
    mx += mx.T

    adj = torch.tensor(mx, dtype=torch.float32)
    t = EpsilonNN(threshold, set_value=1)
    adj = t(adj)
    return adj
