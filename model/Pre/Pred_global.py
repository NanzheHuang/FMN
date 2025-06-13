import torch
import torch.nn as nn
from model.utils import BaseMLP

EPS = 1e-7


# =============== w dir ===============
class Global_update_Layer(nn.Module):
    def __init__(self, dic):
        super(Global_update_Layer, self).__init__()
        self.E = dic['data']['edge_num']
        self.N = dic['data']['node_num']
        self.C = dic['data']['channel_size']
        self.B = dic['data']['batch_size']
        self.device = dic['data']['device']
        self.head_num = dic['pred']['head_num']
        dim = dic['pred']['out_dim']
        use_bias = dic['pred']['use_bias']
        if use_bias is True:
            self.Q1 = nn.Linear(dim, dim, bias=True)
            self.K1 = nn.Linear(dim, dim, bias=True)
            self.V1 = nn.Linear(dim, dim, bias=True)
            self.Q2 = nn.Linear(dim, dim, bias=True)
            self.K2 = nn.Linear(dim, dim, bias=True)
            self.V2 = nn.Linear(dim, dim, bias=True)
        else:
            self.Q1 = nn.Linear(dim, dim, bias=False)
            self.K1 = nn.Linear(dim, dim, bias=False)
            self.V1 = nn.Linear(dim, dim, bias=False)
            self.Q2 = nn.Linear(dim, dim, bias=False)
            self.K2 = nn.Linear(dim, dim, bias=False)
            self.V2 = nn.Linear(dim, dim, bias=False)
        self.act1 = nn.SiLU()
        self.softmax = nn.Tanh()
        self.tanh = nn.Tanh()

        self.net_v = BaseMLP(dim, dim, 1, activation=nn.SiLU(), dropout=0.4, device=self.device)
        self.net_2h = BaseMLP(dim * 2, dim, dim, activation=nn.SiLU(), dropout=0.4, device=self.device)
        # self.net_x = BaseMLP(dim, dim, 1, activation=nn.SiLU(), dropout=0.4, device=self.device)
        self.net_2x = BaseMLP(dim * 2, dim, 1, activation=nn.SiLU(), dropout=0.4, device=self.device)

        self.dim = dim
        self.to(self.device)

    def forward(self, x, xh, e, eh, v):
        """
        :param x: [C,B,N,3]
        :param xh: [C,B,N,64]
        :param v: [C,B,N,3]
        :return:CB2NN3 CB2NF
        """
        x = x.reshape(self.C, self.B, self.N, 3)
        v = v.reshape(self.C, self.B, self.N, 3)
        xh = xh.reshape(self.C, self.B, self.N, -1)

        # Vel update
        q1 = self.Q1(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
        k1 = self.K1(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
        v1 = self.V1(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
        d1 = torch.sqrt(torch.tensor(q1.size(-1))) + EPS
        w1 = torch.matmul(q1.permute(0, 1, 3, 2, 4), k1.permute(0, 1, 3, 4, 2)) / d1  # (N,128)*(128,N) [C,B,2,N,N]
        w1 = self.tanh(w1)
        dir = self.get_dir(x, e)
        res_v = w1.unsqueeze(3) * dir @ v1.permute(0, 1, 3, 2, 4).unsqueeze(3)  # [C,B,2,3,N,N]@[C,B,2,3,N,feat_dim]
        res_v = res_v.permute(0, 1, 4, 3, 2, 5).reshape(self.C, self.B, self.N, 3, -1)  # [C,B,N,3,f]
        g_delta_v = self.net_v(res_v).squeeze()     # [C,B,N,3]

        # X update
        q2 = self.Q2(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
        k2 = self.K2(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
        v2 = self.V2(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
        d2 = torch.sqrt(torch.tensor(q2.size(-1))) + EPS
        w2 = torch.matmul(q2.permute(0, 1, 3, 2, 4), k2.permute(0, 1, 3, 4, 2)) / d2  # (N,128)*(128,N) [C,B,2,N,N]
        w2 = self.softmax(w2) * d2
        res_x = w2 @ v2.permute(0, 1, 3, 2, 4)              # [C,B,2,N,feat_dim]
        res_x = res_x.permute(0, 1, 3, 2, 4).reshape(self.C, self.B, self.N, -1)

        g_delta_x = (v + g_delta_v) * self.net_2x(torch.cat((xh, res_x), dim=-1))
        g_h = self.net_2h(torch.cat((xh, res_x), dim=-1))

        g_delta_x = g_delta_x.reshape(self.C, self.B*self.N, 3)
        g_delta_v = g_delta_v.reshape(self.C, self.B*self.N, 3)
        g_h = g_h.reshape(self.C, self.B*self.N, -1)

        return g_delta_x, g_h


    def get_dir(self, x, e):
        idx0, idx1 = e[0][:self.E], e[1][:self.E]
        dir = torch.zeros((self.C, self.B, self.N, self.N, 3), dtype=torch.float32, device=self.device)
        dir[:, :, idx0, idx1, :] = x[:, :, idx1, :] - x[:, :, idx0, :]     # [C,B,N,N,3]
        return dir.unsqueeze(2).permute(0, 1, 2, 5, 3, 4)     # [C,B,1,N,N,3] -> [C,B,1,3,N,N]


# =============== w/o dir ===============
# class Global_update_Layer(nn.Module):
#     def __init__(self, dic):
#         super(Global_update_Layer, self).__init__()
#         self.E = dic['data']['edge_num']
#         self.N = dic['data']['node_num']
#         self.C = dic['data']['channel_size']
#         self.B = dic['data']['batch_size']
#         self.device = dic['data']['device']
#         self.head_num = dic['pred']['head_num']
#         dim = dic['pred']['out_dim']
#         use_bias = dic['pred']['use_bias']
#         if use_bias is True:
#             self.Q1 = nn.Linear(dim, dim, bias=True)
#             self.K1 = nn.Linear(dim, dim, bias=True)
#             self.V1 = nn.Linear(dim, dim, bias=True)
#             self.Q2 = nn.Linear(dim, dim, bias=True)
#             self.K2 = nn.Linear(dim, dim, bias=True)
#             self.V2 = nn.Linear(dim, dim, bias=True)
#         else:
#             self.Q1 = nn.Linear(dim, dim, bias=False)
#             self.K1 = nn.Linear(dim, dim, bias=False)
#             self.V1 = nn.Linear(dim, dim, bias=False)
#             self.Q2 = nn.Linear(dim, dim, bias=False)
#             self.K2 = nn.Linear(dim, dim, bias=False)
#             self.V2 = nn.Linear(dim, dim, bias=False)
#         self.act1 = nn.SiLU()
#         self.softmax = nn.Tanh()
#         self.tanh = nn.Tanh()
#
#         self.net_v = BaseMLP(dim, dim, 1, activation=nn.SiLU(), dropout=0.4, device=self.device)
#         self.net_2h = BaseMLP(dim * 2, dim, dim, activation=nn.SiLU(), dropout=0.4, device=self.device)
#         self.net_2x = BaseMLP(dim * 2, dim, 3, activation=nn.SiLU(), dropout=0.4, device=self.device)
#
#         self.dim = dim
#         self.to(self.device)
#
#     def forward(self, x, xh, e, eh, v):
#         """
#         :param x: [C,B,N,3]
#         :param xh: [C,B,N,64]
#         :param v: [C,B,N,3]
#         :return:CB2NN3 CB2NF
#         """
#         x = x.reshape(self.C, self.B, self.N, 3)
#         v = v.reshape(self.C, self.B, self.N, 3)
#         xh = xh.reshape(self.C, self.B, self.N, -1)
#
#         # X update
#         q2 = self.Q2(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
#         k2 = self.K2(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
#         v2 = self.V2(xh).reshape(self.C, self.B, self.N, self.head_num, -1)
#         d2 = torch.sqrt(torch.tensor(q2.size(-1))) + EPS
#         w2 = torch.matmul(q2.permute(0, 1, 3, 2, 4), k2.permute(0, 1, 3, 4, 2)) / d2  # (N,128)*(128,N) [C,B,2,N,N]
#         w2 = self.softmax(w2) * d2
#         res_x = w2 @ v2.permute(0, 1, 3, 2, 4)              # [C,B,2,N,feat_dim]
#         res_x = res_x.permute(0, 1, 3, 2, 4).reshape(self.C, self.B, self.N, -1)
#
#         g_delta_x = self.net_2x(torch.cat((xh, res_x), dim=-1))
#         g_h = self.net_2h(torch.cat((xh, res_x), dim=-1))
#
#         g_delta_x = g_delta_x.reshape(self.C, self.B*self.N, 3)
#         g_h = g_h.reshape(self.C, self.B*self.N, -1)
#
#         return g_delta_x, g_h




