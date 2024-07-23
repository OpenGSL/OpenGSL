from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, JumpingKnowledge, SAGEConv, GINConv, GATConv
from torch_geometric.data import Batch
from torch_geometric.typing import (
    OptPairTensor,
    OptTensor,
    Adj,
    Size
)
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.norm import BatchNorm, LayerNorm, GraphNorm
from torch_sparse import SparseTensor, matmul
from torch_scatter import segment_csr
from torch_geometric.utils import cumsum, scatter, is_torch_sparse_tensor


class AttentiveLayer(nn.Module):

    def __init__(self, d, **kwargs):
        super(AttentiveLayer, self).__init__()
        self.d = d
        self.w = nn.Parameter(torch.ones(d))

    def reset_parameters(self):
        self.w = nn.Parameter(self.w.new_ones(self.d))

    def forward(self, x):
        return x @ torch.diag(self.w)


class AttentiveEncoder(nn.Module):
    '''
    Attentive encoder used in `"Towards Unsupervised Deep Graph Structure Learning" <https://arxiv.org/abs/2201.06367>`_ paper.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    d : int
        Dimensions of input features.
    activation: str
        Specify the activation function used.
    '''

    def __init__(self, n_layers, d, activation='F.relu', **kwargs):
        super(AttentiveEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(AttentiveLayer(d))
        self.activation = eval(activation)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj=None):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.

        Returns
        -------
        x : torch.tensor
            Encoded features.
        '''
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (len(self.layers) - 1):
                x = self.activation(x)
        return x


class MLPEncoder(nn.Module):
    '''
    Adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py

    Parameters
    ----------
    n_feat : int
        Dimensions of input features.
    n_hidden : int
        Dimensions of hidden representations.
    n_class : int
        Dimensions of output representations.
    n_layers : int
        Number of layers.
    dropout : float
        Dropout rate.
    use_bn : bool
        Whether to use batch norm.
    activation: str
        Specify the activation function used.
    '''
    def __init__(self, n_feat, n_hidden, n_class, n_layers, dropout=.5, use_bn=True, activation='F.relu', **kwargs):
        super(MLPEncoder, self).__init__()
        self.n_feat = n_feat
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activation = eval(activation)
        if n_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(n_feat, n_class))
        else:
            self.lins.append(nn.Linear(n_feat, n_hidden))
            self.bns.append(BatchNorm(n_hidden))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(n_hidden, n_hidden))
                self.bns.append(BatchNorm(n_hidden))
            self.lins.append(nn.Linear(n_hidden, n_class))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def param_init(self):
        # used by sublime
        for layer in self.lins:
            layer.weight = nn.Parameter(torch.eye(self.n_feat))

    def forward(self, x, **kwargs):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        Returns
        -------
        x : torch.tensor
            Encoded features.
        '''
        if isinstance(x, Batch):
            x= x.x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.5, n_linear=1, bias=True, act='F.relu',
                 weight_initializer=None, bias_initializer=None, **kwargs):
        super(GraphConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.ModuleList()
        self.mlp.append(Linear(in_channels, out_channels, bias=bias, weight_initializer=weight_initializer, bias_initializer=bias_initializer))
        for i in range(n_linear-1):
            self.mlp.append(Linear(out_channels, out_channels, bias=bias, weight_initializer=weight_initializer, bias_initializer=bias_initializer))
        self.dropout = dropout
        self.act = eval(act)

    def reset_parameters(self):
        for layer in self.mlp:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, adj: Union[SparseTensor, torch.Tensor]):

        for i in range(len(self.mlp)-1):
            x = self.mlp[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)

        if isinstance(adj, SparseTensor):
            x = matmul(adj, x)
        elif isinstance(adj, torch.Tensor):
            x = torch.mm(adj, x)
            # x = matmul(SparseTensor.from_torch_sparse_coo_tensor(adj),x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


class GraphConvolutionDiagLayer(nn.Module):
    '''
    A GCN convolution layer of diagonal matrix multiplication
    '''
    def __init__(self, input_size):
        super(GraphConvolutionDiagLayer, self).__init__()
        self.input_size = input_size
        self.W = torch.nn.Parameter(torch.ones(input_size))

    def reset_parameters(self):
        self.W = torch.nn.Parameter(self.W.new_ones(self.input_size))

    def forward(self, x, adj):
        x = x @ torch.diag(self.W)
        if isinstance(adj, SparseTensor):
            x = matmul(adj, x)
        elif isinstance(adj, torch.Tensor):
            x = torch.mm(adj, x)
        return x


class GCNDiagEncoder(nn.Module):
    '''
    Encoder used in `"Graph-Revised Convolutional Network" <https://arxiv.org/abs/1911.07123>`_ paper.

    Parameters
    ----------
    n_layers: int
        Number of layers.
    d : int
        Dimensions of input features.
    activation: str
        Specify the activation function used.
    '''
    def __init__(self, n_layers, d, activation='F.tanh'):
        super(GCNDiagEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GraphConvolutionDiagLayer(d))
        self.activation = eval(activation)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        Returns
        -------
        x : torch.tensor
            Encoded features.

        '''
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != (len(self.layers) - 1):
                x = self.activation(x)
        return x


class GNNEncoder_OpenGSL(nn.Module):
    '''
    GNN encoder.
    TODO to be fused with GNNEncoder_PYG in future versions

    Parameters
    ----------
    n_feat : int
        Dimensions of input features.
    n_hidden : int
        Dimensions of hidden representations.
    n_class : int
        Dimensions of output representations.
    n_layers : int
        Number of layers.
    dropout : float
        Dropout rate.
    input_dropout : float
        Dropout rate on the infut at first. Only used when `input_layer` is `True`.
    norm : str
        Specify batchnorm or layernorm. The operation won't be done if set to `None`.
    n_linear : int
        Number of linear layers in a GCN layer.
    act : str
        Specify the activation function used.
    input_layer : bool
        Whether to set a seperate linear layer for input.
    output_layer: bool
        Whether to set a seperate linear layer for output.
    weight_initializer: str
        Specify the way to initialize the weights.
    bias_initializer: str
        Specify the way to initialize the bias.
    bias : bool
        Whether to add bias to linear transform in GCN.
    '''

    def __init__(self, n_feat, n_class, n_hidden, n_layers=2, dropout=0.5, input_dropout=0.0, norm=None, n_linear=1,
                 act='F.relu', input_layer=False, output_layer=False, weight_initializer=None, bias_initializer=None,
                 bias=True, residual=False, jk=None, n_layers_output=1, **kwargs):

        super(GNNEncoder_OpenGSL, self).__init__()

        self.n_feat = n_feat
        self.nclass = n_class
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.jk = None
        if jk in ['cat', 'max', 'lstm']:
            self.jk = JumpingKnowledge(jk, n_hidden, n_layers)
            self.output_layer = True
        self.n_linear = n_linear
        self.dropout = dropout
        self.residual = residual
        if self.residual:
            assert self.input_layer and self.output_layer
        if norm is None:
            norm = {'flag': False}
        self.norm_flag = norm['flag']
        if self.norm_flag:
            self.norm_type = norm['norm_type']
            self.norm_first = norm['norm_first'] if 'norm_first' in norm else False
            self.supports_norm_batch = self.norm_type in ['LayerNorm', 'GraphNorm']
            norm_layer = eval(self.norm_type)
            norm_kwargs = norm['norm_kwargs'] if 'norm_kwargs' in norm else dict()
        self.act = eval(act)
        if self.input_layer:
            self.input_linear = nn.Linear(in_features=n_feat, out_features=n_hidden)
            self.input_drop = nn.Dropout(input_dropout)
        if self.output_layer:
            n_hidden_output = n_hidden * n_layers if jk == 'cat' else n_hidden
            self.output_linear = MLPEncoder(n_feat=n_hidden_output, n_hidden=n_hidden, n_class=n_class,
                                            n_layers=n_layers_output)
            if self.norm_flag and self.norm_first:
                self.output_normalization = norm_layer(in_channels=n_hidden_output, **norm_kwargs)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = n_feat
            else:
                in_hidden = n_hidden
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = n_class
            else:
                out_hidden = n_hidden
            self.convs.append(GraphConvolutionLayer(in_hidden, out_hidden, dropout, n_linear, act=act, weight_initializer=weight_initializer, bias_initializer=bias_initializer, bias=bias))
            if self.norm_flag:
                if self.norm_first:
                    self.norms.append(norm_layer(in_channels=in_hidden, **norm_kwargs))
                else:
                    self.norms.append(norm_layer(in_channels=out_hidden, **norm_kwargs))

    def reset_parameters(self):
        if self.input_layer:
            self.input_linear.reset_parameters()
        if self.output_layer:
            self.output_linear.reset_parameters()
        for layer in self.convs:
            layer.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, adj=None, return_mid=False):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        return_mid : bool
            Whether to return hidden representations.
        Returns
        -------
        x : torch.tensor
            Encoded features.
        mid : torch.tensor
            hidden representations. Returned when `return_mid` is True.

        '''
        xs = []
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)
        for i, layer in enumerate(self.convs):
            if self.norm_flag and self.norm_first:
                x_res = layer(self.norms[i](x), adj)
            else:
                x_res = layer(x, adj)
            if i < self.n_layers - 1 or self.output_layer:
                if self.norm_flag and (not self.norm_first):
                    x_res = self.norms[i](x_res)
                x_res = self.act(x_res)
                x_res = F.dropout(x_res, p=self.dropout, training=self.training)
            x = x_res + x if self.residual else x_res
            xs.append(x)
        x = self.jk(xs) if self.jk else x
        if self.output_layer:
            mid = x
            if self.norm_flag and self.norm_first:
                x = self.output_normalization(x)
            x = self.output_linear(x)
        else:
            mid = xs[-2]
        if return_mid:
            return mid, x.squeeze(1)
        else:
            return x.squeeze(1)


class GNNEncoder(nn.Module):
    # 各GNN的pyg版本
    # 注意GAT不支持传入edge weight！
    '''
    GNN encoder.

    Parameters
    ----------
    n_feat : int
        Dimensions of input features.
    n_hidden : int
        Dimensions of hidden representations.
    n_class : int
        Dimensions of output representations.
    n_layers : int
        Number of layers.
    dropout : float
        Dropout rate.
    input_dropout : float
        Dropout rate on the infut at first. Only used when `input_layer` is `True`.
    norm : str
        Specify batchnorm or layernorm. The operation won't be done if set to `None`.
    n_linear : int
        Number of linear layers in a GCN layer.
    spmm_type: int
        Specify the multiply funtion used between adj and x.
    act : str
        Specify the activation function used.
    input_layer : bool
        Whether to set a seperate linear layer for input.
    output_layer: bool
        Whether to set a seperate linear layer for output.
    weight_initializer: str
        Specify the way to initialize the weights.
    bias_initializer: str
        Specify the way to initialize the bias.
    bias : bool
        Whether to add bias to linear transform in GCN.
    '''
    def __init__(self, n_feat, n_class, n_hidden, n_layers=2, dropout=0.5, input_dropout=0.0, norm=None, n_linear=1,
                 act='F.relu', input_layer=False, output_layer=False, bias=True, pool=None, residual=False, jk=None,
                 n_layers_output=1, backbone='gcn', **kwargs):

        super(GNNEncoder, self).__init__()

        self.n_feat = n_feat
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.jk = None
        if jk in ['cat', 'max', 'lstm']:
            self.jk = JumpingKnowledge(jk, n_hidden, n_layers)
            self.output_layer = True
        self.n_linear = n_linear
        self.dropout = dropout
        self.pool = pool
        self.residual = residual
        self.backbone = backbone
        assert self.backbone in ['gcn', 'sage', 'gin']
        if self.backbone == 'gcn':
            conv_layer = GCNConv
        elif self.backbone == 'sage':
            conv_layer = SAGEConvPlus
        elif self.backbone == 'gat':
            conv_layer = GATConv
        elif self.backbone == 'gin':
            conv_layer = GINConvPlus
        else:
            raise NotImplementedError
        if self.residual:
            assert self.input_layer and self.output_layer
        if norm is None:
            norm = {'flag':False, 'norm_type':None}
        self.norm_flag = norm['flag'] and norm['norm_type']
        if self.norm_flag:
            self.norm_type = norm['norm_type']
            self.supports_norm_batch = self.norm_type in ['LayerNorm', 'GraphNorm']
            norm_layer = eval(self.norm_type)
            norm_kwargs = norm['norm_kwargs'] if 'norm_kwargs' in norm else dict()
        self.act = eval(act)
        if self.input_layer:
            self.input_linear = nn.Linear(in_features=n_feat, out_features=n_hidden)
            self.input_drop = nn.Dropout(input_dropout)
        if self.output_layer:
            if jk == 'cat':
                self.output_linear = MLPEncoder(n_feat=n_hidden * n_layers, n_hidden=n_hidden, n_class=n_class, n_layers=n_layers_output)
            else:
                self.output_linear = MLPEncoder(n_feat=n_hidden, n_hidden=n_hidden, n_class=n_class, n_layers=n_layers_output)
        self.convs = nn.ModuleList()
        if self.norm_flag:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = n_feat
            else:
                in_hidden = n_hidden
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = n_class
            else:
                out_hidden = n_hidden
            if self.backbone == 'gin':
                mlp = MLPEncoder(in_hidden, out_hidden, out_hidden, n_layers=2, dropout=0, use_bn=False)
                self.convs.append(conv_layer(mlp))
            else:
                self.convs.append(conv_layer(in_hidden, out_hidden, bias=bias))
            if self.norm_flag:
                self.norms.append(norm_layer(in_channels=out_hidden, **norm_kwargs))

    def reset_parameters(self):
        if self.input_layer:
            self.input_linear.reset_parameters()
        if self.output_layer:
            self.output_linear.reset_parameters()
        for layer in self.convs:
            layer.reset_parameters()
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, x=None, edge_index: torch.Tensor = None, edge_attr=None, batch=None, get_cls=True, return_before_pool=False):
        batch_size = batch.max()+1 if batch is not None else 1
        xs = []
        if edge_index.is_sparse:
            edges = SparseTensor.from_torch_sparse_coo_tensor(edge_index)
        else:
            edges = SparseTensor.from_edge_index(edge_index=edge_index, edge_attr=edge_attr, sparse_sizes=(x.shape[0], x.shape[0]))
        if self.input_layer:
            x = self.input_linear(x)
            x = self.act(x)
            x = self.input_drop(x)
        for i, layer in enumerate(self.convs):
            if self.backbone == 'gat':
                x1 = layer(x=x, edge_index=edges)
            else:
                x1 = layer(x=x, edge_index=edges)
            if i < self.n_layers - 1 or self.output_layer:
                if self.norms:
                    if self.supports_norm_batch:
                        x1 = self.norms[i](x1, batch, batch_size)
                    else:
                        x1 = self.norms[i](x1)
                x1 = self.act(x1)
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
            if self.jk:
                xs.append(x1)
            x = x1 + x if self.residual else x1
        x = self.jk(xs) if self.jk else x
        if self.pool:
            x_pooled = global_pool(x, batch, self.pool)
        else:
            x_pooled = x
        if self.output_layer and get_cls:
            z = self.output_linear(x_pooled)
        else:
            z = x_pooled
        if return_before_pool:
            return z, x
        else:
            return z


class APPNPEncoder(nn.Module):
    '''
    APPNP encoder from `"Predict then Propagate: Graph Neural Networks meet Personalized PageRank" <https://arxiv.org/abs/1810.05997>`_ paper.

    Parameters
    ----------
    in_channels : int
        Dimensions of input features.
    hidden_channels : int
        Dimensions of hidden representations.
    out_channels : int
        Dimensions of output representations.
    dropout : float
        Dropout rate.
    K : int
        Number of propagations.
    alpha : float
        Teleport probability.
    spmm_type: int
        Specify the multiply funtion used between adj and x.
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1, spmm_type=0):
        super(APPNPEncoder, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.K = K
        self.alpha = alpha
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj, return_mid=False):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        return_mid : bool
            Whether to return hidden representations.
        Returns
        -------
        z : torch.tensor
            Encoded features.
        x : torch.tensor
            hidden representations. Returned when `return_mid` is True.

        '''
        # adj is normalized and sparse
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        z = x
        for i in range(self.K):
            z = (1-self.alpha)*self.spmm(adj,z)+self.alpha*x
        if return_mid:
            return z, z.squeeze(1)
        else:
            return z.squeeze(1)


class GINEncoder(nn.Module):
    '''
    GIN encoder from `"How Powerful are Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Parameters
    ----------
    n_feat : int
        Dimensions of input features.
    n_hidden : int
        Dimensions of hidden representations.
    n_class : int
        Dimensions of output representations.
    n_layers : int
        Number of layers.
    mlp_layers : int
        Number of MLP layers.
    learn_eps : bool
        Whether to set eps as learned parameters.
    spmm_type: int
        Specify the multiply funtion used between adj and x.
    '''
    def __init__(self, n_feat, n_hidden, n_class, n_layers=3, mlp_layers=1, learn_eps=True, spmm_type=0):
        super(GINEncoder, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.n_layers))
        self.spmm = [torch.spmm, torch.sparse.mm][spmm_type]

        self.mlps = nn.ModuleList()
        if n_layers == 1:
            self.mlps.append(MLPEncoder(n_feat, n_hidden, n_class, mlp_layers, 0))
        else:
            self.mlps.append(MLPEncoder(n_feat, n_hidden, n_hidden, mlp_layers, 0))
            for layer in range(self.n_layers - 2):
                self.mlps.append(MLPEncoder(n_hidden, n_hidden, n_hidden, mlp_layers, 0))
            self.mlps.append(MLPEncoder(n_hidden, n_hidden, n_class, mlp_layers, 0))

    def reset_parameters(self):
        for layer in self.mlps:
            layer.reset_parameters()

    def forward(self, x, adj, return_mid=False):
        '''
        Parameters
        ----------
        x : torch.tensor
            Input features.
        adj : torch.tensor
            Input adj.
        return_mid : bool
            Whether to return hidden representations.
        Returns
        -------
        z : torch.tensor
            Encoded features.
        x : torch.tensor
            hidden representations. Returned when `return_mid` is True.

        '''
        if self.learn_eps:
            for i in range(self.n_layers - 1):
                x = self.mlps[i]((1 + self.eps[i]) * x + self.spmm(adj, x))
                x = F.relu(x)
            z = self.mlps[-1]((1 + self.eps[-1]) * x + self.spmm(adj, x))
        else:
            for i in range(self.n_layers - 1):
                x = self.mlps[i](x + self.spmm(adj, x))
                x = F.relu(x)
            z = self.mlps[-1](x + self.spmm(adj, x))

        if return_mid:
            return x, z.squeeze(1)
        else:
            return z.squeeze(1)


class GINConvPlus(GINConv):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(self, x, edge_index, size=None, edge_weight=None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)   # add edge_weight

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SAGEConvPlus(SAGEConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs
    ):
        super(SAGEConvPlus, self).__init__(in_channels, out_channels, aggr, normalize, root_weight, project, bias, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


def global_pool(x: Tensor, batch: Tensor, reduce='mean') -> Tensor:
    ones = torch.ones_like(batch)
    ptr = cumsum(scatter(ones, batch))
    return segment_csr(x, ptr, reduce=reduce)
