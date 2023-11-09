import dataset as md
import torch
import model
import time
from tqdm import tqdm
#import  args
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch.nn.functional as F
import torch_geometric
import argparse
import json
def compute_loss_para(adj):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm
def get_acc(adj_rec, adj_label):
    labels_all = adj_label.reshape(-1).long()
    preds_all = (adj_rec > 0.5).reshape(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy
def get_scores(edges_pos,edges_neg, adj_rec): #edges_neg
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos.t():
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg.t():
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data)) , preds_neg , np.zeros(len(preds_neg))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

# 获取用户输入参数
# python train.py -dataset Cora -model VGAE -inputDim 1433 -hiddenDim 32 -latentDim 16 -epoch 200 -rate 0.01 --use_feature False
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='Cora', help='dataset')
parser.add_argument('-model', type=str, default='VGAE', help='model')
parser.add_argument('-inputDim', type=int, default=1433, help='inputDim')
parser.add_argument('-hiddenDim', type=int, default=32, help='hiddenDim')
parser.add_argument('-latentDim', type=int, default=16, help='latentDim')
parser.add_argument('-epoch', type=int, default=200, help='num_epoch')
parser.add_argument('-rate', type=float, default=0.01, help='learning_rate')
parser.add_argument('-use_feature', type=bool, default=False, help='use_feature')
parser.add_argument('-help', type=bool, default=False, help='help')
args = parser.parse_args().__dict__
if args.get('help'):
    parser.print_help()
    exit()

finalROC = []
finalAP = []

def train(i):
    myDataset = md.MyDataset(args.get('dataset'))
    myDataset.splitDataset()
    data = myDataset.data
    edges = data.train_pos_edge_index

    myModel = None
    if args.get('model') == 'VGAE':
        myModel = model.VGAE(
            inputDim=args.get('inputDim'),
            hiddenDim=args.get('hiddenDim'),
            latentDim=args.get('latentDim'),
        )
    elif args.get('model')  == 'GAE':
        myModel = model.GAE(
            inputDim=args.get('inputDim'),
            hiddenDim=args.get('hiddenDim'),
            latentDim=args.get('latentDim'),
        )
    else:
        raise NotImplementedError('model {} not implemented'.format(args.get('model')))

    # 优化器
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.get('rate'))
    # 将训练数据转换为邻接矩阵
    adjTrain = torch_geometric.utils.to_scipy_sparse_matrix(edges)

    # 转为tensor
    adjTrain = torch.FloatTensor(adjTrain.todense())
    # adjTrain = adjTrain - torch.diag(torch.diag(adjTrain))

    # 计算损失函数参数
    weight_tensor, norm = compute_loss_para(adjTrain)



    adjTest = torch.FloatTensor(torch_geometric.utils.to_scipy_sparse_matrix(data.test_pos_edge_index).todense())
    adj = []
    rocList = []
    apList = []
    X = data.x
    for epoch in tqdm(range(args.get('epoch')), desc='Epoch', unit='epoch', unit_scale=False):
        t = time.time()
        myModel.train()
        #gaeModel.train()
        adj , mean, logstd, z = myModel.forward(X,edges)
        # 如果形状不一致，就转换形状

        if adj.shape != adjTrain.shape:
            #将小的邻接矩阵转换为大的邻接矩阵的形状
            if adj.shape[0] < adjTrain.shape[0]:
                adj = adj.reshape(adjTrain.shape[0],adj.shape[1])
            else:
                adjTrain = adjTrain.reshape(adj.shape[0],adjTrain.shape[1])

        # 计算损失函数
        loss = norm * F.binary_cross_entropy(input=adj.reshape(-1),target = adjTrain.reshape(-1), weight=weight_tensor)
        kl_divergence = None
        if args.get('model') == 'VGAE':
            kl_divergence = -0.5 / adj.size(0) * (1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2)).sum(1).mean()
            loss -= kl_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(adj, adjTrain)
        roc_score, ap_score = get_scores(data.test_pos_edge_index,data.test_neg_edge_index, adj)
        rocList.append(roc_score.item())
        apList.append(ap_score.item())






    test_roc, test_ap = get_scores(data.test_pos_edge_index,data.test_neg_edge_index, adj)
    print(
            "End of training!",
            "test_roc=",
            "{:.5f}".format(test_roc),
            "test_ap=",
            "{:.5f}".format(test_ap),
        )

    rocList.append(test_roc.item())
    apList.append(test_ap.item())
    finalROC.append(test_roc.item())
    finalAP.append(test_ap.item())
    import matplotlib.pyplot as plt

    # 设置画布大小
    plt.figure(figsize=(10, 10))

    plt.plot(range(args.get('epoch')+1), rocList)
    plt.plot(range(args.get('epoch')+1), apList)
    plt.legend(['roc', 'ap'], loc='upper left')
    plt.title(args.get('dataset') + 'Dataset - ' + args.get('model') + 'Model-Times ' + str(i))
    plt.xlabel('epoch')
    plt.ylabel('roc/ap')

    import os
    if not os.path.exists('./rec'):
        os.mkdir('./rec')
    plt.savefig('./rec/' + args.get('dataset') + '-' + args.get('model') + 'UseFeature' + str(i) + '.jpg')



for i in range(10):
    train(i+1)
# 绘制ROC曲线和AP曲线，横坐标为次数，纵坐标为ROC/AP值

# 输出结果的平均值
print('rocList',finalROC)
print('apList',finalAP)
print('rocList mean',np.mean(finalROC))
print('apList mean',np.mean(finalAP))
# 将结果保存到json文件中
import json
with open('./rec/' + args.get('dataset') + '-' + args.get('model') + '-UseFeature' + '.json','w') as f:
    json.dump({'rocList':finalROC,'apList':finalAP,'rocList mean':np.mean(finalROC),'apList mean':np.mean(finalAP)},f)










# End of training! test_roc= 0.90979 test_ap= 0.92022
# End of training! test_roc= 0.90979 test_ap= 0.92022
# End of training! test_roc= 0.95782 test_ap= 0.95659

