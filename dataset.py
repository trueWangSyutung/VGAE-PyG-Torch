from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.utils import train_test_split_edges

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def load_data(name):
    dataset = Planetoid(root='./dataset', name=name)
    dataset = dataset.shuffle()
    return dataset[0]

    
# Load dataset by name using Planetoid
class MyDataset():
    def __init__(self,name):
        self.name = name
        self.data = load_data(name)
    getEdgeIndex = lambda self: self.data.edge_index
    getX = lambda self: self.data.x
    getY = lambda self: self.data.y
    getTrainMask = lambda self: self.data.train_mask
    getValMask = lambda self: self.data.val_mask
    getTestMask = lambda self: self.data.test_mask
    getDatasets = lambda self: self.data
    getDim = lambda self: self.data.num_features

    def getAdj(self):
        adj = torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index)
        return adj

    def splitDataset(self):
        data = self.data
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data)
        self.data = data
        return data

        
        
        


    
