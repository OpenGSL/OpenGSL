from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BMGCN(nn.Module):
    def __init__(self, num_classes, mlp_module, gcn_module, loss_weight, enhance, device):
        super(BMGCN, self).__init__()

        self.loss_weight = loss_weight
        self.mlp = mlp_module
        self.gcn = gcn_module

        bias = np.ones((num_classes, num_classes))
        np.fill_diagonal(bias, enhance)
        self.bias = torch.FloatTensor(bias).to(device)

    def forward(self, feature, adj, idx, label, labels_oneHot, train_idx):
        B = F.softmax(self.mlp(feature))

        H = get_block_matrix(adj, labels_oneHot, B.clone(), train_idx)

        Q = torch.mm(H, H.t())
        Q = Q * self.bias
        Q = Q / torch.sum(Q, dim=1, keepdim=True)

        score = torch.mm(torch.mm(B, Q), B.t()) * adj
        zero_vec = -9e15 * torch.ones_like(score)
        g = torch.where(adj > 0, score, zero_vec)
        g = F.softmax(g, dim=1)

        output = self.gcn(feature, g)
        logits = F.softmax(output, dim=1)

        gcn_loss = F.nll_loss(torch.log(logits[idx]), label)
        mlp_loss = F.nll_loss(torch.log(B[idx]), label)
        final_loss = self.loss_weight[0] * gcn_loss + self.loss_weight[1] * mlp_loss

        return logits, final_loss, H.detach(), Q.detach(), output.detach()


def get_block_matrix(adj, y, soft_y=None, mask=None):
    soft_y[mask] = y[mask]

    H = torch.mm(soft_y.t(), adj)
    H = torch.mm(H, soft_y) / torch.mm(H, torch.ones_like(soft_y))
    return H
