import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.utils import *

class GENE(nn.Module):
    def __init__(self, dic):
        super(GENE, self).__init__()
        self.C = dic['data']['channel_size']
        hid_dim = dic['eq']['hid_dim']
        act = dic['eq']['act']
        device = dic['data']['device']

        self.net = nn.Sequential(
            nn.Linear(9, hid_dim),
            ReturnAct(act),
            nn.Linear(hid_dim, 9)
        )
        self.random_seed = nn.Parameter(torch.randn(self.C, 9), requires_grad=True)  # (c, 9)
        self.bias = nn.Parameter(torch.randn(self.C, 3))
        self.net_bias = nn.Sequential(
                nn.Linear(3, hid_dim // 3),
                ReturnAct(act),
                nn.Linear(hid_dim // 3, 3)
            )
        self.to(device)

    def forward(self):
        """
            h: (B, N, in_node_nf)
            Features are the same cross batches.
        """
        res = self.net(self.random_seed)
        res = res.reshape(self.C, 3, 3)
        bias = self.net_bias(self.bias)

        Q = Schmidt_cross_3vec(res[:, 0:3, :])
        
        cost = self.channel_cost(Q, bias)
        return Q, bias, cost

    def channel_cost(self, Q, bias):
        """ Q:(C, 3, 3)  bias:(C, 3)  w:(C, 1)"""
        miu_dis = 4
        full_graph_edge = torch.combinations(torch.arange(self.C), 2).permute(1, 0)
        distance = F.pairwise_distance(bias[full_graph_edge[0]], bias[full_graph_edge[1]], p=2)
        cost = torch.sqrt((distance-miu_dis)**2)
        cost_mean = torch.mean(cost)
        # cost_var = torch.var(cost)
        return cost_mean   # + cost_var

