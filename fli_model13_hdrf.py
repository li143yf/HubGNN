import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import util.Sampler
import util.Module
import util.transGNN
import util.transGNN2
import util.transGNN4
import util.transGNN9
import csv
import random
import time
from ogb.nodeproppred import DglNodePropPredDataset
import util
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
import pickle
##############################################     实验
# 参数
methods = ["LDG", "Fennel", "Metis", "DBH", "HDRF", "ne", "sne"]  ##点划分
method = methods[5]  ##ne
num_partitions = 10 
num_epochs = 200
batch_size = 1
num_hidden = 128
lr = 0.001
weight_decay =  5e-4
dropout = 0.2

# d_model =128
# nhead =4
# num_encoder_layers = 5
# num_decoder_layers = 6
# --------------------

aveLOSS = []
aveTrain_acc = []
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


node_num = []
edge_num = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# # dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root="/22085400417/dataset4/arxiv"))
# # graph = dataset[0]
# dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root="/22085400417/22414/obgdata"))
# graph = dataset[0]
dataset = dgl.data.FlickrDataset(raw_dir='/22085400417/dataset4/FlickrDataset')
graph = dataset[0]
##graph = dgl.remove_self_loop(graph)  
###graph = dgl.add_self_loop(graph)
num_edges = graph.num_edges()
num_nodes = graph.num_nodes()
print("rg节点数", num_nodes)
print("rg边数", num_edges)

with open("/22085400417/22414/415/ama_coa_fen/compare/experience/fli_huafen_su/Flickr_part10_hdrf_ture_2jin_commndID", 'rb') as f:
       common_nodes_per_set = pickle.load(f)

degrees= (graph.in_degrees() + graph.out_degrees()).to(device)

# with open("/22085400417/arxhdrf50_commndID", 'rb') as f:
#        common_nodes_per_set = pickle.load(f)

# for i, nodes in enumerate(common_nodes_per_set):
#     print(f"集合 {i+1} 的公共节点: {nodes}")


#model = util.transGNN.TransSAGE1(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes,dropout, d_model, nhead, num_encoder_layers, num_decoder_layers).cuda()

model = util.transGNN9.TransSAGE9(
    in_feats=graph.ndata["feat"].shape[1],
    n_hidden= num_hidden,
    n_classes=dataset.num_classes,
    dropout=dropout,
    d_model=128,
    d_ff=512,
    d_k=64,
    d_v=64,
    n_layers=1,
    n_heads=2
).cuda()

#model = util.Module.SAGE1(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes,dropout).cuda()
##model = util.Module.Graph_Conv(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

dataloader = dgl.dataloading.DataLoader(  #####边划分
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler.Edge_partition_sampler_id(
        graph,
        num_partitions,  ###huafen_blosk
        cache_path=f'/22085400417/22414/415/ama_coa_fen/compare/experience/fli_huafen_su/fli_hdrf10_idlist',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    ),
    device="cuda",
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)  ####训练验证测试用相同的划分方法

eval_dataloader = dgl.dataloading.DataLoader(  #####边划分
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler.Edge_partition_sampler_id(
        graph,
        num_partitions,  ####CoraGraphDataset   ##"DBH", "HDRF","reverse_HDRF"
        cache_path=f'/22085400417/22414/415/ama_coa_fen/compare/experience/fli_huafen_su/fli_hdrf10_idlist',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
        ###"train_mask", "val_mask", "test_mask"
    ),
    device="cuda",
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)  ####训练验证测试用相同的划分方法

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

  # 按行进行L2范数归一化
def l2_normalize_rows(matrix):
    # 计算每行的L2范数
    norms = torch.norm(matrix, p=2, dim=1, keepdim=True)
    # 除以每行的L2范数，进行归一化
    return matrix / norms

# 按行进行标准化
def standardize_rows(matrix):
    # 计算每行的均值
    mean = torch.mean(matrix, dim=1, keepdim=True)
    # 计算每行的标准差
    std = torch.std(matrix, dim=1, keepdim=True)
    # 减去均值并除以标准差，进行标准化
    return (matrix - mean) / std

# durations存放每次训练的时间，后面计算平均值和标准差
durations = []
k = int(10590)
torch.manual_seed(0)
x1 = torch.randn(k, num_hidden).to(device)
layer_norm = torch.nn.LayerNorm(num_hidden).to(device)
x1 = layer_norm(x1)

for e in range(num_epochs):
    print(f"epoch: {e + 1}")
    # 训练部分
    t0 = time.time()
    model.train()
    i = 0
    Loss = []
    Acc = []


    for it, sgid in enumerate(dataloader):  ###训练\
        sg = sgid[0]
        print("sgid", sgid[1])
        # commnd_id = common_nodes_per_set[sgid[1]]
        # commnd_id = sorted(list(commnd_id))
        # commnd_id = torch.tensor(commnd_id).to(device)
        commnd_id = torch.tensor(sorted(common_nodes_per_set[sgid[1]])).to(device)
        
        print(f"sg节点数 { sg.num_nodes()}")
        print(f"commnd_id {len(commnd_id)}")
        # k = int(num_nodes * 0.30)
        # if it == 0 :
        #     x1 = torch.randn(len(commnd_id), num_hidden).to(device)
        i += 1
        # print("sg",sg)
        x = sg.ndata["feat"]
        y = sg.ndata["label"]
        m = sg.ndata["train_mask"].bool()  ###.bool(): 将提取到的节点特征转换为布尔类型。这意味着只有为 True 的节点才会被选中，而 False 的节点将被忽略掉
        print("sg_num_nodes",sg.num_nodes())
        print("len(commnd_id)",len(commnd_id))
        y_hat, la1 = model(sg, x, x1,commnd_id,degrees,k,device)
        print("la1",la1.shape)
        # print("sg",sg)
        ###loss = torch.nn.MultiLabelMarginLoss()(y_hat[m], y[m])  ###mulabel
        loss = F.cross_entropy(y_hat[m], y[m])  ###muclass
        x1 = la1.detach()
        opt.zero_grad()
        loss.backward()
        opt.step()
        Loss.append(loss.item())


        acc = MF.accuracy(y_hat[m], y[m], task="multiclass",
                          num_classes=dataset.num_classes)  ###task="multiclass" 表示多类分类任务，num_classes=dataset.num_classes 表示类别的数量
        ##acc = MF.accuracy(y_hat[m], y[m], task="multilabel", num_labels=dataset.num_classes)
        Acc.append(acc.item())
        mem = torch.cuda.max_memory_allocated() / 1000000  ###获取 CUDA 设备当前的最大内存使用量，并将其转换为以 MB 为单位的值
        # Acc为训练集的精度

        print(f"Loss {loss.item():.4f} | Acc {acc.item():.4f} | Peak Mem {mem:.2f}MB")  ###打印训练损失值、训练集精度和最大内存使用量
    x1_infor= x1
    print("every_epoch_loss", Loss)
    print("every_epoch_traacc", Acc)
    aveLOSS.append(f"{sum(Loss) / i:.8f}")  ##每一轮最后一个loss
    aveTrain_acc.append(f"{sum(Acc) / i:.8f}")

    # tt - to为一轮训练的时间###一个epoch时间
    tt = time.time()
    print(f"time: {tt - t0:.2f}s")
    durations.append(tt - t0)


    ###################################
    ###评估部分
    model.eval()  ###每个epoch评估一次    在sg里面评估测试
    with torch.no_grad():
        # 预测的标签, val为验证集， test为测试集
        val_preds, test_preds = [], []
        # 真实的标签
        val_labels, test_labels = [], []
        y_hat_tensor = torch.empty(num_nodes, dataset.num_classes).to(device)
        val_false_tensor = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        test_false_tensor = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        y_tensor = torch.empty(num_nodes,dtype=torch.long).to(device)
        for it, sgid in enumerate(eval_dataloader):  ###验证集测试集很多个batch
            sg = sgid[0]
            # print("sgid", sgid[1])
            commnd_id = torch.tensor(sorted(common_nodes_per_set[sgid[1]])).to(device)
            x = sg.ndata["feat"]  # x为特征向量矩阵
            y = sg.ndata["label"] 
            # y为分类标签
            # print("y",y)
            m_val = sg.ndata["val_mask"].bool()
            # print("m_val",m_val)
            m_test = sg.ndata["test_mask"].bool()
            y_hat, x1_infor = model.inference(sg, x, x1_infor,commnd_id,degrees,k,device)
            y_hat = F.log_softmax(y_hat, dim=1)
            print("y_hat",y_hat.shape)
            print("sg.num_nodes()",sg.num_nodes())
            
            node_yuan_id  = sg.ndata["_ID"]
            print("node_yuan_id",len(node_yuan_id))
            for i, index in enumerate(node_yuan_id):
                y_hat_tensor[index, :y_hat.shape[1]] += y_hat[i]
                val_false_tensor[index] = m_val[i]
                test_false_tensor[index] = m_test[i]
                y_tensor[index] = y[i] # 如果目标tensor是Long类型

        #     val_preds.append(y_hat[m_val])
        #     val_labels.append(y[m_val])
        #     test_preds.append(y_hat[m_test])
        #     test_labels.append(y[m_test])

        # # 把张量的列表合成一个张量
        # val_preds = torch.cat(val_preds, 0)  ###所有batch的按行拼接
        # val_labels = torch.cat(val_labels, 0)
        # test_preds = torch.cat(test_preds, 0)
        # test_labels = torch.cat(test_labels, 0)

        val_acc = MF.accuracy(y_hat_tensor[val_false_tensor], y_tensor[val_false_tensor], task="multiclass", num_classes=dataset.num_classes)
        test_acc = MF.accuracy(y_hat_tensor[test_false_tensor], y_tensor[test_false_tensor], task="multiclass", num_classes=dataset.num_classes)
        print(f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}")
        # val_acc = compute_acc(y_hat_tensor[val_false_tensor], y_tensor[val_false_tensor])
        # test_acc = compute_acc(y_hat_tensor[test_false_tensor], y_tensor[test_false_tensor])    
        Val_acc.append(val_acc.item())
        Test_acc.append(test_acc.item())

        val_preds = y_hat_tensor[val_false_tensor].cpu().numpy()
        val_labels = y_tensor[val_false_tensor].cpu().numpy()
        test_preds = y_hat_tensor[test_false_tensor].cpu().numpy()
        test_labels = y_tensor[test_false_tensor].cpu().numpy()
        val_preds = np.argmax(val_preds, axis=1)
        test_preds = np.argmax(test_preds, axis=1)
            # 计算每种 F1 分数
        val_f1_micro = f1_score(val_preds, val_labels, average='micro')
        val_f1_macro = f1_score(val_preds, val_labels, average='macro')
        val_f1_weighted = f1_score(val_preds, val_labels, average='weighted')

        test_f1_micro = f1_score(test_preds, test_labels, average='micro')
        test_f1_macro = f1_score(test_preds, test_labels, average='macro')
        test_f1_weighted = f1_score(test_preds, test_labels, average='weighted')
        Val_f1_micro.append(val_f1_micro.item())
        Test_f1_micro.append(test_f1_micro.item())
      
        Val_f1_macro.append(val_f1_macro.item())
        Test_f1_macro.append(test_f1_macro.item())
      
        Val_f1_weighted.append(val_f1_weighted.item())
        Test_f1_weighted.append(test_f1_weighted.item())

        # y_hat_tensor[val_false_tensor] = y_hat_tensor[val_false_tensor].cpu().numpy()
        # y_tensor[val_false_tensor] =  y_tensor[val_false_tensor].cpu().numpy()
        #     test_preds = test_preds.cpu().numpy()
        #     test_labels = test_labels.cpu().numpy()
        #     val_preds = np.argmax(val_preds, axis=1)
        #     test_preds = np.argmax(test_preds, axis=1)

        # val_f1_micro = f1_score(y_hat_tensor[val_false_tensor], y_tensor[val_false_tensor], average='micro')
        # val_f1_macro = f1_score(y_hat_tensor[val_false_tensor], y_tensor[val_false_tensor], average='macro')
        # val_f1_weighted = f1_score(y_hat_tensor[val_false_tensor], y_tensor[val_false_tensor], average='weighted')

        # test_f1_micro = f1_score(y_hat_tensor[test_false_tensor], y_tensor[test_false_tensor], average='micro')
        # test_f1_macro = f1_score(y_hat_tensor[test_false_tensor], y_tensor[test_false_tensor], average='macro')
        # test_f1_weighted = f1_score(y_hat_tensor[test_false_tensor], y_tensor[test_false_tensor], average='weighted')
    #     # 评估部分
    # model.eval()  ###每个epoch评估一次    在sg里面评估测试
    # with torch.no_grad():
    #     model = model.cpu()
    #     x = graph.ndata["feat"]
    #     y = graph.ndata["label"]
    #     m_val = graph.ndata["val_mask"].bool()
    #     m_test = graph.ndata["test_mask"].bool()
    #     y_hat = model.inference_full(graph, x)

    #     val_acc = MF.accuracy(y_hat[m_val], y[m_val], task="multiclass", num_classes=dataset.num_classes)
    #     test_acc = MF.accuracy(y_hat[m_test], y[m_test], task="multiclass", num_classes=dataset.num_classes)
    #     print(f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}")
    #     Val_acc.append(val_acc.item())
    #     Test_acc.append(test_acc.item())
    # model.train()
    # model = model.to(device)

# 打印最终结果
print("Final results:")

best_val_index = Val_acc.index(max(Val_acc))
best_test_acc = Test_acc[best_val_index]
print(f"最佳验证集准确率: {max(Val_acc)}")
print(f"对应的测试集准确率: {best_test_acc}")
print(f"最佳的测试集准确率: {max(Test_acc)}")
print(f"Average training time per epoch: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")

best_val_index = Val_f1_micro.index(max(Val_f1_micro))
best_test_acc = Test_f1_micro[best_val_index]
print(f"最佳验证集f1_micro: {max(Val_f1_micro)}")
print(f"对应的测试集f1_micro: {best_test_acc}")
print(f"最佳的测试集f1_micro: {max(Test_f1_micro)}")

best_val_index = Val_f1_macro.index(max(Val_f1_macro))
best_test_acc = Test_f1_macro[best_val_index]
print(f"最佳验证集f1_macro: {max(Val_f1_macro)}")
print(f"对应的测试集f1_macro: {best_test_acc}")
print(f"最佳的测试集f1_macro: {max(Test_f1_macro)}")

best_val_index = Val_f1_weighted.index(max(Val_f1_weighted))
best_test_acc = Test_f1_weighted[best_val_index]
print(f"最佳验证集f1_weighted: {max(Val_f1_weighted)}")
print(f"对应的测试集f1_weighted: {best_test_acc}")
print(f"最佳的测试集f1_weighted: {max(Test_f1_weighted)}")
