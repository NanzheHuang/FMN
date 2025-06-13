import torch
import torch.nn as nn
from model.utils import BaseMLP

EPS = 1e-7


# =============== w dir ===============
def aggregate(message, row_index, result_shape, mod=4, aggr='sum', mask=None):
    C, BE, dim = result_shape
    if mod == 4:
        result = message.new_full(result_shape, 0).unsqueeze(-1).repeat(1, 1, 1, 3)
        row_index = row_index.reshape(1, -1, 1, 1).repeat(C, 1, dim, 3)         # [C,BE,256,3]
    else:
        result = message.new_full(result_shape, 0)
        row_index = row_index.reshape(1, -1, 1).repeat(C, 1, dim)           # [C,BE,256]
    result.scatter_add_(1, row_index, message)                              # [C,BE,256,3]
    if aggr == 'sum':
        pass
    elif aggr == 'mean':
        if mod == 4:
            count = message.new_full(result_shape, 0).unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            count = message.new_full(result_shape, 0)
        ones = torch.ones_like(message)
        if mask is not None:
            ones = ones * mask.unsqueeze(-1)
        count.scatter_add_(1, row_index, ones)
        result = result / count.clamp(min=1)
    else:
        raise NotImplementedError('Unknown aggregation method:', aggr)
    return result


class Local_update_Layer(nn.Module):
    def __init__(self, dic):
        super(Local_update_Layer, self).__init__()
        self.dim = dic['pred']['out_dim']
        device = dic['data']['device']
        self.scalar_net = BaseMLP(self.dim * 2 + 1, self.dim, self.dim, nn.SiLU(), device, dropout=0.2)
        self.node_net1 = BaseMLP(self.dim, self.dim, 3, nn.SiLU(), device)
        self.node_net2 = BaseMLP(self.dim, self.dim, 1, nn.SiLU(), device)
        self.node_net3 = BaseMLP(self.dim * 2, self.dim, self.dim, nn.SiLU(), device, dropout=0.2)
        self.to(device)

    def forward(self, x, xh, e):
        idx0, idx1 = e[0], e[1]

        rij = x[:, idx0, :] - x[:, idx1, :]                             # [C,BE,3]
        hij = torch.cat((xh[:, idx0, :], xh[:, idx1, :]), dim=-1)       # [C,BE,512]
        dir = rij / torch.norm(rij, dim=-1, keepdim=True)               # [C,BE,3]

        scalar = torch.sum(rij ** 2, dim=-1, keepdim=True)              # [C,BE,1]
        scalar = torch.cat((scalar, hij), dim=-1)                       # [C,BE,513]
        mij = self.scalar_net(scalar)                                   # [C,BE,256]
        mij = mij.unsqueeze(-1) * dir.unsqueeze(-2)                     # [C,BE,256,3]

        new_xh_nb = aggregate(message=mij, row_index=idx0, result_shape=xh.shape, mod=4, aggr='mean')
        new_xh_nb_norm = torch.norm(new_xh_nb, dim=-1, keepdim=False)

        x = self.node_net1(xh) + self.node_net2(new_xh_nb.transpose(-2, -1)).squeeze()
        new_h = self.node_net3(torch.cat([xh, new_xh_nb_norm], dim=-1))

        return x, new_h, new_xh_nb_norm


# =============== w/o dir ===============
# def aggregate(message, row_index, result_shape, mod=4, aggr='sum', mask=None):
#     C, BE, dim = result_shape
#     if mod == 4:
#         result = message.new_full(result_shape, 0).unsqueeze(-1).repeat(1, 1, 1, 3)
#         row_index = row_index.reshape(1, -1, 1, 1).repeat(C, 1, dim, 3)         # [C,BE,256,3]
#     else:
#         result = message.new_full(result_shape, 0)
#         row_index = row_index.reshape(1, -1, 1).repeat(C, 1, dim)           # [C,BE,256]
#     result.scatter_add_(1, row_index, message)                              # [C,BE,256,3]
#     if aggr == 'sum':
#         pass
#     elif aggr == 'mean':
#         if mod == 4:
#             count = message.new_full(result_shape, 0).unsqueeze(-1).repeat(1, 1, 1, 3)
#         else:
#             count = message.new_full(result_shape, 0)
#         ones = torch.ones_like(message)
#         if mask is not None:
#             ones = ones * mask.unsqueeze(-1)
#         count.scatter_add_(1, row_index, ones)
#         result = result / count.clamp(min=1)
#     else:
#         raise NotImplementedError('Unknown aggregation method:', aggr)
#     return result
#
#
# class Local_update_Layer(nn.Module):
#     def __init__(self, dic):
#         super(Local_update_Layer, self).__init__()
#         self.dim = dic['pred']['out_dim']
#         device = dic['data']['device']
#         self.scalar_net = BaseMLP(self.dim * 2 + 1, self.dim, self.dim, nn.SiLU(), device, dropout=0.2)
#         self.node_net1 = BaseMLP(self.dim, self.dim, 3, nn.SiLU(), device)
#         self.node_net2 = BaseMLP(self.dim, self.dim, 3, nn.SiLU(), device)
#         self.node_net3 = BaseMLP(self.dim * 2, self.dim, self.dim, nn.SiLU(), device, dropout=0.2)
#         self.to(device)
#
#     def forward(self, x, xh, e):
#         idx0, idx1 = e[0], e[1]
#
#         rij = x[:, idx0, :] - x[:, idx1, :]                             # [C,BE,3]
#         hij = torch.cat((xh[:, idx0, :], xh[:, idx1, :]), dim=-1)       # [C,BE,512]
#
#         scalar = torch.sum(rij ** 2, dim=-1, keepdim=True)              # [C,BE,1]
#         scalar = torch.cat((scalar, hij), dim=-1)                       # [C,BE,513]
#         mij = self.scalar_net(scalar)                                   # [C,BE,256]
#
#         new_xh_nb = aggregate(message=mij, row_index=idx0, result_shape=xh.shape, mod=3, aggr='mean')
#
#         x = self.node_net1(xh) + self.node_net2(new_xh_nb)
#         new_h = self.node_net3(torch.cat([xh, new_xh_nb], dim=-1))
#
#         return x, new_h, new_xh_nb
















# ========= Edit 1 ============
# 加入显示的方向性指导:
# malonaldehyde的效果见好，但未达到SOTA. 结果=12.85；SOTA=12.80
# aspirin效果变差，但依然是SOTA，结果=6.40；SOTA=6.74
# benzene优势可以保持不变
'''
class Local_update_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, device):
        super(Local_update_Layer, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim, self.activation = input_dim, hidden_dim, output_dim, activation
        self.scalar_net = BaseMLP(self.input_dim * 2 + 1, self.hidden_dim, self.output_dim, self.activation, device)
        self.node_net1 = BaseMLP(self.input_dim, self.hidden_dim, 3, self.activation, device)
        self.node_net2 = BaseMLP(self.input_dim, self.hidden_dim, 1, self.activation, device)
        self.node_net3 = BaseMLP(self.hidden_dim * 2, self.hidden_dim, self.hidden_dim, self.activation, device)
        self.to(device)

    def forward(self, x, xh, e):
        idx0, idx1 = e[0], e[1]

        rij = x[:, idx0, :] - x[:, idx1, :]                             # [C,BE,3]
        hij = torch.cat((xh[:, idx0, :], xh[:, idx1, :]), dim=-1)       # [C,BE,512]
        dir = rij / torch.norm(rij, dim=-1, keepdim=True)               # [C,BE,3]

        scalar = torch.sum(rij ** 2, dim=-1, keepdim=True)              # [C,BE,1]
        scalar = torch.cat((scalar, hij), dim=-1)                       # [C,BE,513]
        mij = self.scalar_net(scalar)                                   # [C,BE,256]
        mij = mij.unsqueeze(-1) * dir.unsqueeze(-2)                     # [C,BE,256,3]

        new_xh_nb = aggregate(message=mij, row_index=idx0, result_shape=xh.shape, mod=4, aggr='mean')
        new_xh_nb_norm = torch.norm(new_xh_nb, dim=-1, keepdim=False)
        x = self.node_net1(xh) + self.node_net2(new_xh_nb.transpose(-2, -1)).squeeze()
        new_h = self.node_net3(torch.cat([xh, new_xh_nb_norm], dim=-1))

        return x, new_h, new_xh_nb_norm
'''


# ========= Edit 2 ============
# 去掉显示的方向性指导，只保留隐式方向指导:
# malonaldehyde的效果相比edit1变差，结果=13.1；SOTA=12.80
# aspirin效果相较edit1变好，结果=6.09；SOTA=6.74
# benzene优势可以保持不变
'''
class Local_update_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, device):
        super(Local_update_Layer, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim, self.activation = input_dim, hidden_dim, output_dim, activation
        self.scalar_net = BaseMLP(self.input_dim * 2 + 1, self.hidden_dim, self.output_dim, self.activation, device)
        self.node_net1 = BaseMLP(self.input_dim, self.hidden_dim, 3, self.activation, device)
        self.node_net2 = BaseMLP(self.input_dim, self.hidden_dim, 3, self.activation, device)
        self.node_net3 = BaseMLP(self.hidden_dim * 2, self.hidden_dim, self.hidden_dim, self.activation, device)
        self.to(device)

    def forward(self, x, xh, e):
        idx0, idx1 = e[0], e[1]

        rij = x[:, idx0, :] - x[:, idx1, :]                             # [C,BE,3]
        hij = torch.cat((xh[:, idx0, :], xh[:, idx1, :]), dim=-1)       # [C,BE,512]

        scalar = torch.sum(rij ** 2, dim=-1, keepdim=True)              # [C,BE,1]
        scalar = torch.cat((scalar, hij), dim=-1)                       # [C,BE,513]
        mij = self.scalar_net(scalar)                                   # [C,BE,256]

        new_xh_nb = aggregate(message=mij, row_index=idx0, result_shape=xh.shape, mod=3, aggr='mean')

        x = self.node_net1(xh) + self.node_net2(new_xh_nb)
        new_h = self.node_net3(torch.cat([xh, new_xh_nb], dim=-1))

        return x, new_h, new_xh_nb
'''




