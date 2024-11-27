from typing import *
import os
import torch
import dgl
import random
import numpy as np
import networkx as nx
import json

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
import pickle
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data 
from dgl import AddReverse
from dgl import add_reverse_edges
import util.Sampler
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from collections import Counter
from collections import defaultdict
import time 
from sklearn.metrics import f1_score


def train_model(model_class,
                num_layers,
                train_mask,
                valid_mask,
                test_mask,
                graph,
                node_features,
                node_labels,
                n_features,
                n_labels,
                epoch_num=1000,
                groups=None,
                seed=42,
                device="cuda:0",
                lr=0.001,
                N_P=10,
                B_S=20,
                args=None,
                logger=None):
         
    train_loader = dgl.dataloading.DataLoader(  
    graph,
    torch.arange(N_P).to("cuda"),
    util.Sampler.Edge_partition_sampler_id_2(
        graph,
        N_P,  
        cache_path=f'./dataset/partition/arxiv_hdrf200',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    ),
    device="cuda",
    batch_size=B_S,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=True,)  


                  
    best_valid_acc = 0
    best_test_acc = 0
    best_epoch = 0
    test_acc = 0
    tpr = []
    fpr = []
    model = model_class(n_nodes=graph.ndata["feat"].size(0),
                        in_feats=n_features,
                        out_feats=n_labels,
                        num_layers=num_layers,
                        n_units=args.gnn_hidden_dim,
                        dropout=0.5,
                        args=args).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    Val_acc = []
    Test_acc = []
    Val_f1 = []
    Test_f1 = []
    Val_f1_micro = []
    Test_f1_micro = []      
    Val_f1_macro = []
    Test_f1_macro = []
    Val_f1_weighted = []
    Test_f1_weighted = []              
    for epoch in range(epoch_num):
        model.train()
        total_loss = total_nodes = 0
      
        for it, sgid in enumerate(train_loader):  
            sg = sgid[0]
            opt.zero_grad()
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m = sg.ndata["train_mask"].bool() 
            # sg = dgl.add_self_loop(sg)    ########################################   GCN GAT
            y_hat = model(sg,x)
            loss = F.cross_entropy(y_hat[m], y[m])
            loss.backward() 
            opt.step()
            nodes = m.sum().item()
            total_loss += loss.item() * nodes 
            total_nodes += nodes
        epoch_loss = total_loss / total_nodes
        if (epoch + 1) % 1 == 0:
            val_acc,test_acc,val_f1_micro,test_f1_micro,val_f1_macro,test_f1_macro,val_f1_weighted,test_f1_weighted = evaluate_graph(model,graph,device,args)  
            print("epoch",epoch + 1,"ACC",val_acc,test_acc,"f1_micro",val_f1_micro,test_f1_micro,
                  "f1_macro",val_f1_macro,test_f1_macro,"f1_weighted",val_f1_weighted,test_f1_weighted)
            Val_acc.append(val_acc.item())
            Test_acc.append(test_acc.item())
            Val_f1_micro.append(val_f1_micro.item())
            Test_f1_micro.append(test_f1_micro.item())
      
            Val_f1_macro.append(val_f1_macro.item())
            Test_f1_macro.append(test_f1_macro.item())
      
            Val_f1_weighted.append(val_f1_weighted.item())
            Test_f1_weighted.append(test_f1_weighted.item())

    print("Final results:")

    best_val_index = Val_acc.index(max(Val_acc))
    best_test_acc = Test_acc[best_val_index]
    print(f"val_acc: {max(Val_acc)}")
    print(f"test_acc: {best_test_acc}")
   


    best_val_index = Val_f1_micro.index(max(Val_f1_micro))
    best_test_acc = Test_f1_micro[best_val_index]
    print(f"val_f1_micro: {max(Val_f1_micro)}")
    print(f"test_f1_micro: {best_test_acc}")
  
    best_val_index = Val_f1_macro.index(max(Val_f1_macro))
    best_test_acc = Test_f1_macro[best_val_index]
    print(f"val_f1_macro: {max(Val_f1_macro)}")
    print(f"test_f1_macro: {best_test_acc}")
   

    best_val_index = Val_f1_weighted.index(max(Val_f1_weighted))
    best_test_acc = Test_f1_weighted[best_val_index]
    print(f"val_f1_weighted: {max(Val_f1_weighted)}")
    print(f"test_f1_weighted: {best_test_acc}")
    

    return best_test_acc
   

  
@torch.no_grad()
def evaluate_graph(model,graph,device,args):
    model.eval()
    
    graph = graph.to(device)
    # graph = dgl.add_self_loop(graph)    ################################   GCN  GAT
    total_test_examples = 0
    total_correct = 0
 
    x = graph.ndata["feat"] 
    y = graph.ndata["label"] 
    m_val = graph.ndata["val_mask"].bool() 
    m_test = graph.ndata["test_mask"].bool()
    y_hat = model(graph,x)
    out_val = y_hat[m_val]
    _, indices_val = torch.max(out_val, dim=1)
    labels_val = y[m_val]
    correct_val = torch.sum(indices_val == labels_val)
    num_val_examples = m_val.sum().item()
    acc_val = correct_val / num_val_examples
  
    out_test = y_hat[m_test]
    _, indices_test = torch.max(out_test, dim=1)
    labels_test = y[m_test]
    correct_test = torch.sum(indices_test == labels_test)
    num_test_examples = m_test.sum().item()
    acc_test = correct_test / num_test_examples

    val_preds = y_hat[m_val].cpu().numpy()  
    val_labels = y[m_val].cpu().numpy()
    test_preds = y_hat[m_test].cpu().numpy()
    test_labels =y[m_test].cpu().numpy()
    val_preds = np.argmax(val_preds, axis=1)
    test_preds = np.argmax(test_preds, axis=1)

    val_f1_micro = f1_score(val_preds, val_labels, average='micro')
    val_f1_macro = f1_score(val_preds, val_labels, average='macro')
    val_f1_weighted = f1_score(val_preds, val_labels, average='weighted')

    test_f1_micro = f1_score(test_preds, test_labels, average='micro')
    test_f1_macro = f1_score(test_preds, test_labels, average='macro')
    test_f1_weighted = f1_score(test_preds, test_labels, average='weighted')

    return acc_val,acc_test,val_f1_micro,test_f1_micro,val_f1_macro,test_f1_macro,val_f1_weighted,test_f1_weighted

