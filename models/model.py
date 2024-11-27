import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn.pytorch import GraphConv, GATConv, SGConv, APPNPConv
import types
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import ModuleList
import dgl
from lib.transformer_utilities.transformer_layer import TransformerEncoderLayerVanilla
from lib.transformer_utilities.GroupLinearLayer import GroupLinearLayer
import math


class HubGNN(nn.Module):
    def __init__(self, 
                 n_nodes,
                 in_feats,
                 out_feats,
                 n_units,
                 num_layers=2,
                 dropout=0.1,
                 activation='relu',
                 fc_units=128,
                 args=None):
        super(HubGNN, self).__init__()
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

        self.conv1 = dgl.nn.SAGEConv(in_feats, n_units, "mean")
        self.conv2 = dgl.nn.SAGEConv(n_units + n_units  , out_feats, "mean")
        # self.conv1 = dgl.nn.GraphConv(in_feats, n_units)
        # self.conv2 = dgl.nn.GraphConv(n_units + n_units, out_feats)      
        
        # # Replace GraphConv with GATConv
        # self.conv1 = dgl.nn.GATConv(
        #     in_feats,
        #     n_units,
        #     num_heads= 4,
        #     feat_drop=0,
        #     attn_drop=0,
        #     activation=self.activation,
        #     negative_slope=0.2
        # )
        
        # # For the second conv layer, input features will be n_units * num_heads x2
        # self.conv2 = dgl.nn.GATConv(
        #     n_units * 8,
        #     out_feats,
        #     num_heads=4,
        #     feat_drop=0,
        #     attn_drop=0,
        #     activation=None,
        #     negative_slope=0.2
        # )
                   
        self.memory_layer1 = TransformerEncoderLayerVanilla(args)
        self.layer_norm = torch.nn.LayerNorm(128)        
        self.shared_memory_attention = args.shared_memory_attention
        self.use_topk = args.use_topk
        self.topk = args.topk
        # self.gw_ratio = args.gw_ratio
        self.init_memory = args.init_memory

    def forward(self, sg, x, plot=None):
        x = self.conv1(sg, x)
        # x = x.flatten(1)  # Flatten the heads    ###################  GAT
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x_gw = x.unsqueeze(1)
        if self.memory_layer1.self_attn.memory is not None:
            self.memory_layer1.self_attn.memory = self.memory_layer1.self_attn.memory.detach()
        if self.init_memory:
            self.memory_layer1.self_attn.init_memory(x_gw.size(1), device=x.device)
        x_gw, memory = self.memory_layer1(x_gw, None, memory=self.memory_layer1.self_attn.memory, plot=plot)
        x_gw = x_gw.squeeze(1)
        x = self.layer_norm(x)
        x = torch.cat((x, x_gw), dim=1)
        x = self.conv2(sg, x)
        # x = x.mean(1)  # Average over heads for final prediction    ################   GAT
        return F.log_softmax(x, dim=-1)



                


