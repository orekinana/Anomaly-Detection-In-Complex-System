import numpy as np
import torch
import torch.nn as nn

class GAT(nn.Module):

    def __init__(self, node_num):
        super(GAT, self).__init__()
        self.node_num = node_num
        self.fc = nn.Linear(node_num, node_num)
        self.softmax = nn.Softmax(dim=-1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, adj):
        ones, zeros = torch.ones(self.node_num, self.node_num), torch.zeros(self.node_num, self.node_num)
        adj = self.softmax(adj)
        adj = torch.where(adj > adj.median(), ones, zeros)

        weight = torch.matmul(torch.unsqueeze(x,dim=-1), torch.unsqueeze(x,dim=1))
        weight = self.leakyrelu(self.fc(weight))
        weight = self.softmax(weight)
        weight = weight * adj
        output = torch.matmul(torch.unsqueeze(x,dim=1), weight)
        
        return torch.squeeze(output)