import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch_geometric.nn import GCNConv


class VGAE(nn.Module):
    def __init__(self,inputDim,hiddenDim,latentDim):
        super(VGAE,self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.latentDim = latentDim
        # 三层GCN
        self.base_gcn = GCNConv(self.inputDim, self.hiddenDim, activation=F.relu, allow_zero_in_degree=True)
        self.mean_gcn = GCNConv(self.hiddenDim, self.latentDim, activation=lambda x:x, allow_zero_in_degree=True)
        self.logstd_gcn = GCNConv(self.hiddenDim, self.latentDim, activation=lambda x:x, allow_zero_in_degree=True)

    def encoder(self,X,edge_index):
        hidden = self.base_gcn(X,edge_index)
        self.mean = self.mean_gcn(hidden,edge_index)
        self.logstd = self.logstd_gcn(hidden,edge_index)
        gaussian_noise = torch.randn(X.size(0), self.latentDim)
        # 重采样
        z = self.mean + gaussian_noise * torch.exp(self.logstd)
        return z
    def decoder(self,z):
        adjReconstruct = torch.sigmoid(torch.matmul(z,z.t()))
        #解码器点乘还原邻接矩阵A'
        return adjReconstruct
    def forward(self,X,edge_index):
        z = self.encoder(X,edge_index)
        adjReconstruct = self.decoder(z)
        return adjReconstruct,self.mean,self.logstd,z
    

class GAE(nn.Module):
    def __init__(self,inputDim,hiddenDim,latentDim):
        super(GAE,self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.latentDim = latentDim
        # 两层GCN
        self.base_gcn = GCNConv(self.inputDim, self.hiddenDim, activation=F.relu, allow_zero_in_degree=True)
        self.mean_gcn = GCNConv(self.hiddenDim, self.latentDim, activation=lambda x:x, allow_zero_in_degree=True)
    def encoder(self,X,edge_index):
        hidden = self.base_gcn(X,edge_index)
        self.mean = self.mean_gcn(hidden,edge_index)
        return self.mean
    def decoder(self,z):
        adjReconstruct = torch.sigmoid(torch.matmul(z,z.t()))
        #解码器点乘还原邻接矩阵A'
        return adjReconstruct
    def forward(self,X,edge_index):
        z = self.encoder(X,edge_index)
        adjReconstruct = self.decoder(z)
        return adjReconstruct,self.mean,self.mean,z
    
    

