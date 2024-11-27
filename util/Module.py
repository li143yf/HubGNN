import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import dgl
import dgl.nn as dglnn
from dgl.nn import GCN2Conv

class SAGE1(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes,dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h


class Graph_Conv(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes,dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)

    # sg为子图
    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h

class GCN2(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes,dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCN2Conv(n_hidden, 1, alpha=0.1, lambda_=0.5))
        self.layers.append(GCN2Conv(n_hidden, 2, alpha=0.1, lambda_=0.5))
        self.layers.append(GCN2Conv(n_hidden, 3, alpha=0.1, lambda_=0.5))
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_feats, n_hidden))
        self.lins.append(nn.Linear(n_hidden,n_classes))

    def forward(self, sg, x):
        if self.dropout:
            x = F.dropout(x, p=self.dropout)

        x = x_0 = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout)
        
        for l, layer in enumerate(self.layers):
            x = layer(sg,x, x_0)
            if l != len(self.layers) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout)
        x = self.lins[1](x)
        return x

    def inference(self, sg, x):
        x = x_0 = self.lins[0](x).relu_()
        for l, layer in enumerate(self.layers):
            x = layer(sg,x, x_0)
            if l != len(self.layers) - 1:
                x = x.relu_()
        x = self.lins[1](x)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes,dropout,num_heads,activation):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(
                in_feats,
                n_hidden,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=activation,
                negative_slope=0.2,
            ))
      
        self.layers.append(dglnn.GATConv(
                n_hidden * num_heads,
                n_classes,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=None,
                negative_slope=0.2,
            ))
        

    def forward(self,sg, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(sg, h)
            if l < len(self.layers) - 1:
                h = h.flatten(1)
        return h.mean(1)

    def inference(self, sg, x,dropout):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(sg, h)
            if l < len(self.layers) - 1:
                h = h.flatten(1)
        return h.mean(1)






