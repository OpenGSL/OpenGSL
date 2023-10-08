import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.functional import one_hot
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn.dense.linear import Linear


class SGC(nn.Module):
    def __init__(self, n_feat, n_class, K=2):
        super(SGC, self).__init__()
        self.K = K
        self.W = nn.Linear(n_feat, n_class)
        self.features = None

    def sgc_precompute(self, features, adj):
        for i in range(self.K):
            features = torch.spmm(adj, features)
            self.features = features

    def forward(self, input):
        # adj is normalized and sparse
        # compute feature propagation only once
        x=input[0]
        adj=input[1]
        if self.features is None:
            self.sgc_precompute(x, adj)
        return self.W(self.features).squeeze(1)


class LPA(nn.Module):
    def __init__(self, n_layers=3, alpha=0.9):
        super(LPA, self).__init__()
        self.n_layers = n_layers
        self.alpha = alpha

    def forward(self, input):
        # adj is normalized(without self-loop) and sparse
        # check the format of y
        y = input[0]
        adj = input[1]
        mask = input[2]
        if not len(y.shape) == 2:
            # convert to one hot labels
            y = one_hot(y).float()
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]
        res = (1 - self.alpha) * out
        for _ in range(self.n_layers):
            out = torch.spmm(adj, out)
            out = out * self.alpha + res
            out.clamp_(0., 1.)
        return out


class LINK(nn.Module):
    """ logistic regression on adjacency matrix """

    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, adj):
        logits = self.W(adj)
        return logits.squeeze(1)


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, use_bn=True):
        super(MLP, self).__init__()
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, feats):
        x = feats
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class LINKX(nn.Module):
    """ 
    adapted from 
    a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):
        self.mlpA.reset_parameters()
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()

    def forward(self, input):
        feat=input[0]
        A=input[1]
        xA = self.mlpA(A)
        xX = self.mlpX(feat)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x)

        return x


class APPNP(nn.Module):
    '''
    APPNP Implementation
    Weight decay on the first layer and dropout on adj are not used.
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1, spmm_type=0):
        super(APPNP, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.K = K
        self.alpha = alpha
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, input):
        # adj is normalized and sparse
        x=input[0]
        adj=input[1]
        only_z = input[2] if len(input) > 2 else True
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        z = x
        for i in range(self.K):
            z = (1-self.alpha)*self.spmm(adj,z)+self.alpha*x
        # x = self.prop1(x, edge_index)
        if only_z:
            return z.squeeze(1)
        else:
            return z, z.squeeze(1)


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, dprate=.5, K=10, alpha=.1, init='SGC'):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.dprate = dprate
        self.K = K
        self.alpha = alpha

        assert init in ['SGC', 'PPR', 'NPPR', 'Random']
        if init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = torch.zeros(K+1)
            TEMP[0] = 1.0
        elif init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**torch.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**torch.arange(K+1)
            TEMP = TEMP/torch.sum(torch.abs(TEMP))
        elif init == 'Random':
            # Random
            bound = torch.sqrt(3/(K+1))
            TEMP = torch.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/torch.sum(torch.abs(TEMP))


        self.temp = nn.Parameter(TEMP)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        torch.nn.init.zeros_(self.temp)
        self.temp.data[self.k] = self.alpha*(1-self.alpha)**torch.arange(self.K+1)
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, input):
        # adj is normalized and sparse
        x=input[0]
        adj=input[1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        x = F.dropout(x, p=self.dprate, training=self.training)
        z = x*self.temp[0]
        for i in range(self.K):
            x = torch.spmm(adj,x)
            z = z + self.temp[i+1]*x
        return z.squeeze(1)


class GAT(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layers, n_heads, dropout):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(n_feat, n_hidden, heads=n_heads, dropout=dropout))
        for i in range(n_layers-2):
            self.convs.append(GATConv(n_hidden*n_heads, n_hidden, heads=n_heads, dropout=dropout))
        self.convs.append((GATConv(n_hidden*n_heads, n_class, heads=n_heads, dropout=dropout, concat=False)))
        self.dropout = dropout

    def forward(self, input):
        x = input[0]
        edge_index = input[1]

        x = F.dropout(x, training=self.training, p=self.dropout)

        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = F.elu(x)
        x = self.convs[-1](x, edge_index)
        return x.squeeze(1)


class GIN(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layers=3, mlp_layers=1, learn_eps=True, spmm_type=0):
        super(GIN, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.n_layers))
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]
        
        self.mlps = nn.ModuleList()
        if n_layers == 1:
            self.mlps.append(MLP(n_feat, n_hidden, n_class, mlp_layers, 0))
        else:
            self.mlps.append(MLP(n_feat, n_hidden, n_hidden, mlp_layers, 0))
            for layer in range(self.n_layers-2):
                self.mlps.append(MLP(n_hidden, n_hidden, n_hidden, mlp_layers, 0))
            self.mlps.append(MLP(n_hidden, n_hidden, n_class, mlp_layers, 0))
        
    def forward(self, input):
        x = input[0]
        adj = input[1]
        only_z = input[2] if len(input) > 2 else True

        if self.learn_eps:
            for i in range(self.n_layers-1):
                x = self.mlps[i]((1+self.eps[i])*x + self.spmm(adj,x))
                x = F.relu(x)
            z = self.mlps[-1]((1+self.eps[-1])*x + self.spmm(adj,x))
        else:
            for i in range(self.n_layers-1):
                x = self.mlps[i](x + self.spmm(adj,x))
                x = F.relu(x)
            z = self.mlps[-1](x + self.spmm(adj,x))
        
        if only_z:
            return z.squeeze(1)
        else:
            return x, z.squeeze(1)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, n_linear=1, bias=True, spmm_type=1, act='F.relu',
                 last_layer=False, weight_initializer=None, bias_initializer=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.ModuleList()
        self.mlp.append(Linear(in_features, out_features, bias=bias, weight_initializer=weight_initializer, bias_initializer=bias_initializer))
        for i in range(n_linear-1):
            self.mlp.append(Linear(out_features, out_features, bias=bias, weight_initializer=weight_initializer, bias_initializer=bias_initializer))
        self.dropout = dropout
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]
        self.act = eval(act)
        self.last_layer = last_layer


    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        x = self.spmm(adj, input)
        for i in range(len(self.mlp)-1):
            x = self.mlp[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)
        if not self.last_layer:
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, n_layers=5, dropout=0.5, input_dropout=0.0, norm=None, n_linear=1,
                 spmm_type=0, act='F.relu', input_layer=False, output_layer=False, weight_initializer=None,
                 bias_initializer=None, bias=True):

        super(GCN, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.n_linear = n_linear
        if norm is None:
            norm = {'flag':False, 'norm_type':'LayerNorm'}
        self.norm_flag = norm['flag']
        self.norm_type = eval('nn.'+norm['norm_type'])
        self.act = eval(act)
        if input_layer:
            self.input_linear = nn.Linear(in_features=nfeat, out_features=nhid)
            self.input_drop = nn.Dropout(input_dropout)
        if output_layer:
            self.output_linear = nn.Linear(in_features=nhid, out_features=nclass)
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
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = nclass
            else:
                out_hidden = nhid

            self.convs.append(GraphConvolution(in_hidden, out_hidden, dropout, n_linear, spmm_type=spmm_type, act=act,
                                               weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                                               bias=bias))
            if self.norm_flag:
                self.norms.append(self.norm_type(in_hidden))
        self.convs[-1].last_layer = True

    def forward(self, input):
        x=input[0]
        adj=input[1]
        only_z = input[2] if len(input) > 2 else True
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
            if i != self.n_layers - 1:
                mid = x

        if self.output_layer:
            x = self.output_normalization(x)
            x = self.output_linear(x).squeeze(1)
        if only_z:
            return x.squeeze(1)
        else:
            return mid, x.squeeze(1)