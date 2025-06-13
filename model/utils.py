import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def EdgeVec(self, P, edge_idx):
    """ P:(B, N, 3)  edge_idx:(2, B*edge_num)"""
    edge_num = edge_idx.size(-1) // P.size(0)
    edge_idx = edge_idx[:, :edge_num]  # (2, edge_num)
    P_start = P[:, edge_idx[0, :], :]
    P_end = P[:, edge_idx[1, :], :]
    return P_end - P_start

def Schmidt_cross_2vec(vec):
    """ vec: (B, 2, 3)->(B, 3, 3) """
    v1 = vec[:, 0, :]
    v2 = vec[:, 1, :]
    dot = torch.sum(torch.mul(v1, v2), dim=1, keepdim=True) / \
          torch.sum(torch.mul(v1, v1), dim=1, keepdim=True)
    v2 = v2 - v1 * dot
    v3 = torch.cross(v1, v2)  # (B, 3)
    v1 = F.normalize(v1, dim=-1).view(-1, 3, 1)
    v2 = F.normalize(v2, dim=-1).view(-1, 3, 1)
    v3 = F.normalize(v3, dim=-1).view(-1, 3, 1)
    res = torch.cat((v1, v2, v3), -1)
    return res

def Schmidt_cross_3vec(vec):
    """ vec: (C, 3, 3)->(C, 3, 3) """
    a1, b1 = vec[:, 0, :], vec[:, 0, :]
    a2 = vec[:, 1, :]
    a3 = vec[:, 2, :]

    dot0 = torch.sum(torch.mul(b1, a2), dim=1, keepdim=True) / \
           torch.sum(torch.mul(b1, b1), dim=1, keepdim=True)
    b2 = a2 - b1 * dot0

    dot1 = torch.sum(torch.mul(b1, a3), dim=1, keepdim=True) / \
           torch.sum(torch.mul(b1, b1), dim=1, keepdim=True)
    dot2 = torch.sum(torch.mul(b2, a3), dim=1, keepdim=True) / \
           torch.sum(torch.mul(b2, b2), dim=1, keepdim=True)
    b3 = a3 - b1 * dot1 - b2 * dot2

    b1 = F.normalize(b1, dim=-1).view(-1, 3, 1)
    b2 = F.normalize(b2, dim=-1).view(-1, 3, 1)
    b3 = F.normalize(b3, dim=-1).view(-1, 3, 1)
    res = torch.cat((b1, b2, b3), -1)
    return res


def ReturnAct(which_act):
    if which_act == 'SiLU':
        return nn.SiLU()
    elif which_act == 'ReLU':
        return nn.ReLU()


class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation,
                 device='cuda', residual=False, last_act=False, flat=False, dropout=None):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        elif dropout != None:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim)
            )
        self.to(device)

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x