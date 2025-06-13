import torch
from torch import nn

from model.Eq.Rotation import RotoTrans
from model.Pre.Predict import GT
from model.Emb.Emb import Embedding

EPS = 1e-6

class EPPM(nn.Module):
    def __init__(self, dic):
        super(EPPM, self).__init__()
        self.C = dic['data']['channel_size']
        self.B = dic['data']['batch_size']
        self.channel = dic['data']['channel_size']
        device = dic['data']['device']

        # ROTO
        self.ROTO_Module = RotoTrans(dic)
        # EMB
        self.EMB_Module = Embedding(dic)
        # TRANS
        self.TRANS_Module = GT(dic)

        # w of dim C
        self.wc = torch.rand(self.channel, device=device)
        self.wc_net = nn.Sequential(
            nn.Linear(self.channel, 10, bias=False),
            nn.SiLU(),
            nn.Linear(10, self.channel, bias=False),
            nn.Softmax()
        )

        self.to(device)

    def unpack(self, x, x_attr, e, e_attr, v):
        B = self.B
        N = x.size(0) // self.B
        E = e_attr.size(0) // self.B
        x = x.reshape(B, N, -1)  # [BN, 3] -> [B, N, 3]
        v = v.reshape(B, N, -1)  # [BN, 3] -> [B, N, 3]
        x_attr = x_attr.reshape(B, N, -1)  # [BN, 1] -> [B, N, ?]
        e_attr = e_attr.reshape(B, E, -1)  # [BN, 1] -> [B, N, ?]
        return x, x_attr, e, e_attr, v, B, N, E

    def forward(self, x, x_attr, e, e_attr, v=None):
        """go forward: have translation"""
        x, x_attr, e, e_attr, v, B, N, E = self.unpack(x, x_attr, e, e_attr, v)

        # 1. Get Rotation Matrix R
        R, t, cost1 = self.ROTO_Module(x)        # (C,B,3,3)

        # 2. Translate
        x = x.unsqueeze(0).repeat(self.C, 1, 1, 1)      # [C,B,N,d]
        x = x + t

        # 3. Rotation (RP = Q)
        x = torch.matmul(R, x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).reshape(-1, B*N, 3)  # (C, B*N, d)
        if v is not None:
            v = v.unsqueeze(0).repeat(self.C, 1, 1, 1)      # (C,B,N,d)
            v = torch.matmul(R, v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).reshape(-1, B*N, 3)  # (c, B*N, d)
        
        # 4. Prediction
        x_attr = self.EMB_Module(x, x_attr, e, e_attr, v)
        new_x = self.TRANS_Module(x, x_attr, e, e_attr, v)

        # 5. De-Rotation (P = R^T * Q)
        new_x = new_x.reshape(-1, B, N, 3)
        new_x = torch.matmul(R.permute(0, 1, 3, 2), new_x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        # 6. De-Translate
        new_x = new_x - t

        # 7. Aggregation of dim C
        w = self.wc_net(self.wc)
        w = w.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        new_x = (w * new_x).sum(dim=0, keepdim=False)
        # new_x = torch.mean(new_x, dim=0, keepdim=False)

        return new_x.reshape([-1, 3])



