import torch
import torch.nn as nn
import numpy as np
from model.utils import ReturnAct

EPS = 1e-5

# =============== w implicit dir-inf ===============
class Embedding(nn.Module):
    def __init__(self, dic):
        super(Embedding, self).__init__()

        self.E = dic['data']['edge_num']
        self.N = dic['data']['node_num']
        self.B = dic['data']['batch_size']
        self.C = dic['data']['channel_size']
        self.size_eh = dic['data']['edge_attr_size']
        self.size_xh = dic['data']['node_attr_size']
        self.k = dic['emb']['k']
        self.dim = dic['emb']['each_dim']
        self.device = dic['data']['device']
        self.out_dim = dic['pred']['out_dim']

        self.embedding_H = nn.Sequential(
            nn.Linear(self.size_xh + self.k, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.embedding_X = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.embedding_V = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.embedding_F = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.out_dim)
        )
        self.to(dic['data']['device'])

    def forward(self, x, x_attr, e, e_attr, v):
        """
        input:
            x_attr: [B,N,2]
            e_attr: [B,E,5]
            x: [C,BN,3]
            v: [C,BN,3]
        """
        x = x.reshape(self.C, self.B, self.N, 3)
        v = v.reshape(self.C, self.B, self.N, 3)

        abs_val, abs_vec = self.eig(x, e, mod='abstract')
        abs_feat = (abs_val / abs_val.max()).unsqueeze(0) * abs_vec
        abs_feat = abs_feat.unsqueeze(0).repeat(self.B, 1, 1).detach()          # [B,N,k]
        h = torch.cat((x_attr, abs_feat), dim=-1)

        h = h.reshape(self.B * self.N, -1)
        x = x.reshape(self.C, -1, 3)
        v = v.reshape(self.C, -1, 3)

        H = self.embedding_H(h)
        X = self.embedding_X(x)
        V = self.embedding_V(v)  # (B*N, point_f) -> (B*N, 64)
        f_h = torch.cat((H.unsqueeze(0).repeat(self.C, 1, 1), X, V), dim=-1)
        f_h = self.embedding_F(f_h)
        return f_h

    def eig(self, x, e, mod='abs'):
        idx0, idx1 = e[0][0:self.E], e[1][0:self.E]
        if mod == 'abstract':
            adj = torch.zeros([self.N, self.N], dtype=torch.float32, device=self.device)
            adj[idx0, idx1] = 1
            degree = torch.sum(adj, dim=-1, keepdim=False)
            D = torch.diag(degree)
            lap = D - adj
            D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
            lap = D_inv_sqrt @ lap @ D_inv_sqrt
            val, vec = torch.linalg.eigh(lap, UPLO='U')
            val = torch.flip(val, dims=[-1])[:self.k]
            vec = torch.flip(vec, dims=[-1])[:, :self.k]
        else:
            adj = torch.zeros([self.B, self.N, self.N], dtype=torch.float32, device=self.device)
            adj[:, idx0, idx1] = torch.sum((x[:, idx0, :] - x[:, idx1, :]) ** 2, dim=-1, keepdim=False)
            degree = torch.sum(adj, dim=-1, keepdim=True)
            eye_mask = torch.eye(self.N, dtype=torch.float32, device=self.device)
            eye_mask = eye_mask.reshape(1, self.N, self.N).repeat(self.B, 1, 1)
            D = degree.repeat(1, 1, self.N) * eye_mask
            lap = D - adj
            D_inv_sqrt = torch.linalg.inv(D)
            lap = D_inv_sqrt @ lap @ D_inv_sqrt
            val, vec = torch.linalg.eigh(lap, UPLO='U')
            val = torch.flip(val, dims=[-1])[:, :self.k]
            vec = torch.flip(vec, dims=[-1])[:, :, :self.k]
        return val, vec







# =============== w/o implicit dir-inf ===============

# class Embedding(nn.Module):
#     def __init__(self, dic):
#         super(Embedding, self).__init__()
#
#         self.E = dic['data']['edge_num']
#         self.N = dic['data']['node_num']
#         self.B = dic['data']['batch_size']
#         self.C = dic['data']['channel_size']
#         self.size_eh = dic['data']['edge_attr_size']
#         self.size_xh = dic['data']['node_attr_size']
#         self.k = dic['emb']['k']
#         self.dim = dic['emb']['each_dim']
#         self.device = dic['data']['device']
#         self.out_dim = dic['pred']['out_dim']
#
#         self.embedding_H = nn.Sequential(
#             nn.Linear(self.size_xh + self.k, self.dim),
#             nn.SiLU(),
#             nn.Linear(self.dim, self.dim),
#         )
#         self.embedding_F = nn.Sequential(
#             nn.Linear(self.dim, self.dim),
#             nn.SiLU(),
#             nn.Linear(self.dim, self.out_dim)
#         )
#         self.to(dic['data']['device'])
#
#     def forward(self, x, x_attr, e, e_attr, v):
#         """
#         input:
#             x_attr: [B,N,2]
#             e_attr: [B,E,5]
#             x: [C,BN,3]
#             v: [C,BN,3]
#         """
#         abs_val, abs_vec = self.eig(x, e, mod='abstract')
#         abs_feat = (abs_val / abs_val.max()).unsqueeze(0) * abs_vec
#         abs_feat = abs_feat.unsqueeze(0).repeat(self.B, 1, 1).detach()          # [B,N,k]
#         h = torch.cat((x_attr, abs_feat), dim=-1)
#
#         h = h.reshape(self.B * self.N, -1)
#
#         H = self.embedding_H(h)
#         f_h = H.unsqueeze(0).repeat(self.C, 1, 1)
#         f_h = self.embedding_F(f_h)
#         return f_h
#
#     def eig(self, x, e, mod='abs'):
#         idx0, idx1 = e[0][0:self.E], e[1][0:self.E]
#         if mod == 'abstract':
#             adj = torch.zeros([self.N, self.N], dtype=torch.float32, device=self.device)
#             adj[idx0, idx1] = 1
#             degree = torch.sum(adj, dim=-1, keepdim=False)
#             D = torch.diag(degree)
#             lap = D - adj
#             D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
#             lap = D_inv_sqrt @ lap @ D_inv_sqrt
#             val, vec = torch.linalg.eigh(lap, UPLO='U')
#             val = torch.flip(val, dims=[-1])[:self.k]
#             vec = torch.flip(vec, dims=[-1])[:, :self.k]
#         else:
#             adj = torch.zeros([self.B, self.N, self.N], dtype=torch.float32, device=self.device)
#             adj[:, idx0, idx1] = torch.sum((x[:, idx0, :] - x[:, idx1, :]) ** 2, dim=-1, keepdim=False)
#             degree = torch.sum(adj, dim=-1, keepdim=True)
#             eye_mask = torch.eye(self.N, dtype=torch.float32, device=self.device)
#             eye_mask = eye_mask.reshape(1, self.N, self.N).repeat(self.B, 1, 1)
#             D = degree.repeat(1, 1, self.N) * eye_mask
#             lap = D - adj
#             D_inv_sqrt = torch.linalg.inv(D)
#             lap = D_inv_sqrt @ lap @ D_inv_sqrt
#             val, vec = torch.linalg.eigh(lap, UPLO='U')
#             val = torch.flip(val, dims=[-1])[:, :self.k]
#             vec = torch.flip(vec, dims=[-1])[:, :, :self.k]
#         return val, vec
