import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, n_linear=1, bias=True, spmm_type=1, act='relu'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_features, out_features, bias=bias))
        for i in range(n_linear-1):
            self.mlp.append(nn.Linear(out_features, out_features, bias=bias))
        self.dropout = dropout
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]
        self.act = eval('F.'+act) if not act=='identity' else lambda x: x


    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        x = self.spmm(adj, input)
        for i in range(len(self.mlp)-1):
            x = self.mlp[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class JKNet(nn.Module):

    def __init__(self, nfeat, nhid, nclass, n_layers=5, dropout=0.5, input_dropout=0.0, norm=None, n_linear=1, spmm_type=0, act='relu', general='concat', input_layer=True, output_layer=True):

        super(JKNet, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.n_linear = n_linear
        self.norm_flag = norm['flag']
        self.norm_type = eval("nn."+norm['norm_type'])
        self.general_aggregation = eval('self.'+general)
        self.act = eval('F.'+act) if not act == 'identity' else lambda x: x
        if input_layer:
            self.input_linear = nn.Linear(in_features=nfeat, out_features=nhid)
            self.input_drop = nn.Dropout(input_dropout)
        if output_layer:
            self.output_linear = nn.Linear(in_features=nhid * n_layers, out_features=nclass)
            self.output_normalization = self.norm_type(nhid)
        self.convs = nn.ModuleList()
        if self.norm_flag:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = nfeat
            else:
                in_hidden = nhid

            out_hidden = nhid

            self.convs.append(GraphConvolution(in_hidden, out_hidden, dropout, n_linear, spmm_type=spmm_type, act=act))
            if self.norm_flag:
                self.norms.append(self.norm_type(in_hidden))

        if (general == 'concat'):
            self.last_layer = nn.Linear(nhid * n_layers, nclass)
        else :
            self.last_layer = nn.Linear(nhid, nclass) 
        if (general == 'LSMT'):
            self.lstm = nn.LSTM(nhid, (n_layers * nhid) // 2, bidirectional=True, batch_first=True)
            self.attn = nn.Linear(2 * ((n_layers * nhid) // 2), 1)


    def forward(self, input):
        x=input[0]
        adj=input[1]
        only_z=input[2]
        layer_outputs = []
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)

        for i, layer in enumerate(self.convs):
            if self.norm_flag:
                x_res = self.norms[i](x)
                x_res = layer(x_res, adj)
                x = x + x_res
            else:
                x = layer(x,adj)
            layer_outputs.append(x)
        mid = self.general_aggregation(layer_outputs)
        if self.output_layer:
            x = self.output_normalization(x)
        x = self.last_layer(mid).squeeze(1)
        if only_z:
            return x
        else:
            return mid, x

    def concat(self, layer_outputs):
        return torch.cat(layer_outputs, dim=1)
    
    def maxpool(self, layer_outputs):
        return torch.max(torch.stack(layer_outputs, dim=0), dim=0)[0]
    
    def LSMT(self, layer_outputs):
        x = torch.stack(layer_outputs, dim=1)
        alpha, _ = self.lstm(x)
        alpha = self.attn(alpha).squeeze(-1)
        alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
        return (x * alpha).sum(dim=1)
