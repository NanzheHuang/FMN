import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean

eps = 1e-7
# class MemBank(nn.Module):
#     def __init__(self, dic):
#         super(MemBank, self).__init__()
#         dim = dic['pred']['out_dim']
#         self.device = dic['data']['device']
#
#         self.mask_net = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.SiLU(),
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )
#         self.memory = None
#
#         self.to(self.device)
#
#     def forward(self, h, i, layer):
#         if i == 0:
#             self.memory = h
#             return h
#         elif i == layer-1:
#             return h
#         else:
#             res = self.mask_net(h)
#             mask = res > 0.5
#             get = torch.zeros_like(mask, device=self.device)
#             get[mask] = 1.0
#             get[~mask] = 0.0
#             get_h = h * get + self.memory * (-1*get+1)
#             self.memory = get_h
#         return get_h


class MemBank(nn.Module):
    def __init__(self, dic):
        super(MemBank, self).__init__()
        dim = dic['pred']['out_dim']
        self.device = dic['data']['device']

        self.mask_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.memory = None

        self.to(self.device)

    def forward(self, h, i, layer):
        if i == 0:
            self.memory = h
            return h
        # elif i == layer-1:
        #     return h
        else:
            get = self.mask_net(h)
            get_h = h * get + self.memory * (1.0 - get)
            self.memory = get_h
        return get_h






