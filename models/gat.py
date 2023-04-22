import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv
from collections import Iterable
import dgl

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.heads = heads

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, False, None))

    def forward(self, blocks, inputs):
        h = inputs

        if not isinstance(blocks, Iterable):
            g = blocks
            blocks = [g] * (self.num_layers + 1)
        elif len(blocks) == 1:
            g = blocks[0]
            blocks = [g] * (self.num_layers + 1)

        for l in range(self.num_layers):
            h = self.gat_layers[l](blocks[l], h).flatten(1)

        # output projection
        logits = self.gat_layers[-1](blocks[-1], h).mean(1)
        return logits.squeeze(1)

    def inference(self, g, x, device, batch_size, num_workers):

        i = 0
        for l, layer in enumerate(self.gat_layers):
            feat_shape = self.num_hidden * self.heads[l] if l != len(self.gat_layers)-1 else self.num_classes
            y = torch.zeros(g.num_nodes(), feat_shape)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)

                if l != len(self.gat_layers) - 1:
                    h = h.flatten(1)
                else:
                    h = h.mean(1)

                y[output_nodes] = h.cpu()

            x = y
            i += 1
        
        return y.squeeze(1)

