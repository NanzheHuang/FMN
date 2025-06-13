import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from model.Eq.Generator import GENE
from model.utils import *


# ============= w FA ==================
class RotoTrans(nn.Module):
    def __init__(self, dic):
        super(RotoTrans, self).__init__()
        self.C = dic['data']['channel_size']
        device = dic['data']['device']

        self.weight_C = nn.Parameter(torch.randn(self.C, 1), requires_grad=True)
        self.GENE = GENE(dic)

        self.to(device)

    def forward(self, P):
        """
            P: (B, N, d)
            edge_index: (2, B*130)
        """
        Q_basis, Q_cent, cost1 = self.GENE()    # Q_basia (C,3,3), bias (C,3)
        Q_basis = Q_basis.unsqueeze(0).repeat(P.size(0), 1, 1, 1)  # -> (B,c,3,3)   列向量

        # 1. Translate para
        P_cent = P.mean(dim=1, keepdim=True)                                # [B, 1, 3]
        P_cent = P_cent.unsqueeze(1).repeat(1, self.C, 1, 1)                # [B, C, 1, 3]
        Q_cent = Q_cent.unsqueeze(0).unsqueeze(2).expand(P_cent.size())     # [B, C, 1, 3]
        t = Q_cent - P_cent
        t = t.transpose(0, 1)                                               # [C, B, 1, 3]

        # 2. R
        P_basis = self.get_P_basis(P)                               # (B,edge_num,3) -> (B,2,3)     row_vec
        P_basis = Schmidt_cross_2vec(P_basis)                       # (B,2,3) -> (B,3,3)            col_vec
        P_basis = P_basis.unsqueeze(1).repeat(1, self.C, 1, 1)      # (B,3,3) -> (B,C,3,3)
        P_basis_T = P_basis.permute([0, 1, 3, 2])
        R = torch.matmul(Q_basis, P_basis_T)                        # (B,C,3,3)
        R = R.transpose(0, 1)                             # (C,B,3,3)

        return R, t, cost1     # (C,B,3,3) (C,B,1,3)

    def get_P_basis(self, x):
        '''
        :param x:   [B, N, 3]
        :return:    [B, 3, 3]
        '''
        x = x - x.mean(dim=1, keepdim=True)             # Centralization
        x = x * x.norm(dim=1, keepdim=True) ** 2        # Scaling
        cov = (x.transpose(-2, -1) @ x) / (x.size(1) - 1)
        val, vec = torch.linalg.eigh(cov, UPLO='U')
        vec = Schmidt_cross_3vec(vec.transpose(-2, -1))
        return vec






# ============= w/o FA ==================
# class RotoTrans(nn.Module):
#     def __init__(self, dic):
#         super(RotoTrans, self).__init__()
#         self.C = dic['data']['channel_size']
#         device = dic['data']['device']
#         self.device = device
#
#         self.weight_C = nn.Parameter(torch.randn(self.C, 1), requires_grad=True)
#         self.GENE = GENE(dic)
#
#         self.to(device)
#
#     def forward(self, P):
#         """
#             P: (B, N, d)
#             edge_index: (2, B*130)
#         """
#         Q_basis, Q_cent, cost1 = self.GENE()    # Q_basia (C,3,3), bias (C,3)
#
#
#         Q_basis = torch.tensor([[[1, 0, 0],
#                                 [0, 1, 0],
#                                 [0, 0, 1]]], dtype=Q_basis.dtype, device=self.device)
#         Q_cent = torch.tensor([[0, 0, 0]], dtype=Q_cent.dtype, device=self.device)
#
#
#         Q_basis = Q_basis.unsqueeze(0).repeat(P.size(0), 1, 1, 1)  # -> (B,c,3,3)   列向量
#
#         # 1. Translate para
#         P_cent = P.mean(dim=1, keepdim=True)                                # [B, 1, 3]
#         P_cent = P_cent.unsqueeze(1).repeat(1, self.C, 1, 1)                # [B, C, 1, 3]
#         Q_cent = Q_cent.unsqueeze(0).unsqueeze(2).expand(P_cent.size())     # [B, C, 1, 3]
#         t = Q_cent - P_cent
#         t = t.transpose(0, 1)                                               # [C, B, 1, 3]
#
#         # 2. R
#         P_basis = self.get_P_basis(P)                               # (B,edge_num,3) -> (B,2,3)     row_vec
#         P_basis = Schmidt_cross_2vec(P_basis)                       # (B,2,3) -> (B,3,3)            col_vec
#         P_basis = P_basis.unsqueeze(1).repeat(1, self.C, 1, 1)      # (B,3,3) -> (B,C,3,3)
#         P_basis_T = P_basis.permute([0, 1, 3, 2])
#         R = torch.matmul(Q_basis, P_basis_T)                        # (B,C,3,3)
#         R = R.transpose(0, 1)                             # (C,B,3,3)
#
#         return R, t, cost1     # (C,B,3,3) (C,B,1,3)
#
#     def get_P_basis(self, x):
#         '''
#         :param x:   [B, N, 3]
#         :return:    [B, 3, 3]
#         '''
#         x = x - x.mean(dim=1, keepdim=True)             # Centralization
#         x = x * x.norm(dim=1, keepdim=True) ** 2        # Scaling
#         cov = (x.transpose(-2, -1) @ x) / (x.size(1) - 1)
#         val, vec = torch.linalg.eigh(cov, UPLO='U')
#         vec = Schmidt_cross_3vec(vec.transpose(-2, -1))
#         return vec


