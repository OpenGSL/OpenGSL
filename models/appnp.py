import torch.nn as nn
from torch_geometric.nn.conv import APPNP as APPNPConv
import torch.nn.functional as F
import torch


class APPNP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1):
        super(APPNP, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop1 = APPNPConv(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x.squeeze(1)

class MyAPPNP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1):
        super(MyAPPNP, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.K = K
        self.alpha = alpha

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj):
        # adj is normalized and sparse
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        z = x
        for i in range(self.K):
            z = (1-self.alpha)*torch.spmm(adj,z)+self.alpha*x
        # x = self.prop1(x, edge_index)
        return z.squeeze(1)


if __name__ == '__main__':
    model = MyAPPNP(128,16,8)
    print(model)