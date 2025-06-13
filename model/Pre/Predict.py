import torch
import torch.nn as nn

from model.Pre.MemBank import MemBank
from model.Pre.Pred_global import Global_update_Layer
from model.Pre.Pred_local import Local_update_Layer
from model.utils import BaseMLP


class GT(nn.Module):
    def __init__(self, dic):
        super(GT, self).__init__()
        self.layer_num = dic['pred']['layer_num']
        self.N = dic['data']['node_num']
        self.C = dic['data']['channel_size']
        self.B = dic['data']['batch_size']
        dim = dic['pred']['out_dim']
        head_num = dic['pred']['head_num']
        batch_size = dic['data']['batch_size']
        device = dic['data']['device']

        self.mb = MemBank(dic)
        self.Global_update = nn.ModuleList([Global_update_Layer(dic) for _ in range(self.layer_num)])
        self.Local_update = nn.ModuleList([Local_update_Layer(dic) for _ in range(self.layer_num)])
        self.h_update = nn.ModuleList([BaseMLP(dim, dim, dim, nn.SiLU(), device) for _ in range(self.layer_num)])
        self.norm = nn.BatchNorm1d(self.B)

        self.to(device)

    def forward(self, x, xh, e, eh, v):
        for i in range(self.layer_num):
            xh = self.mb(xh, i, self.layer_num)

            # Global Update
            temp_g_x, temp_g_h = self.Global_update[i](x, xh, e, eh, v)

            # Local Update
            temp_l_x, temp_l_h, temp_l_h_i = self.Local_update[i](x, xh, e)

            # Update X
            x = x + temp_g_x.reshape(temp_g_x.shape[0], -1, 3) + temp_l_x
            xh = torch.stack([temp_l_h, temp_l_h_i, temp_g_h.reshape(temp_g_h.shape[0], -1, temp_g_h.shape[-1])], dim=-1)
            xh = self.h_update[i](xh.max(-1)[0])

        return x

