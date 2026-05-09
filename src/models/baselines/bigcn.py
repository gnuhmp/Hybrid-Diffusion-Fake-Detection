import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
import copy as cp

import torch
from torch.utils.data import random_split
try:
    from torch_scatter import scatter_mean
except ImportError:
    from torch_geometric.nn import global_mean_pool as scatter_mean_fallback
    def scatter_mean(x, batch, dim=0):
        return scatter_mean_fallback(x, batch)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel

try:
    from src.utils.data_loader import *
    from src.utils.eval_helper import *
except ImportError:
    pass

class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)
        rootindex = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
        batch_size = max(data.batch) + 1

        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x

class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)

        rootindex = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x

class BiGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes=2):
        super(BiGCN, self).__init__()
        self.TDrumorGCN = TDrumorGCN(num_features, hidden_dim, hidden_dim)
        self.BUrumorGCN = BUrumorGCN(num_features, hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear((hidden_dim+hidden_dim) * 2, num_classes)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')
    parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--TDdroprate', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--BUdroprate', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=45, help='maximum number of epochs')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
    parser.add_argument('--feature', type=str, default='profile', help='feature type, [profile, spacy, bert, content]')
    
    args = parser.parse_args()
    print("Test mode running...")