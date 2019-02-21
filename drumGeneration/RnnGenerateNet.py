import torch.nn as nn
import torch.nn.functional as F

import torch
# class RnnGenerateNet(nn.Module):
#
#     def __init__(self,num_features=9,gru_hidden_size=64,batch_size=4096):
#         super(RnnGenerateNet, self).__init__()
#
#         self.num_features=num_features
#         self.gru_hidden_size=gru_hidden_size
#         self.num_directions=1
#         self.seq_len=16
#         self.gru_out_dim = self.seq_len * self.gru_hidden_size * self.num_directions
#         #
#         self.gru0 = torch.nn.GRU(
#             input_size=self.num_features,
#             num_layers=1,
#             hidden_size=16,
#             bias=True,
#             batch_first=True,
#             bidirectional=True)
#
#
#
#         self.gru1 = torch.nn.GRU(
#             input_size=16*2,
#             num_layers=2,
#             hidden_size=9,
#             bias=True,
#             batch_first=True,
#             bidirectional=False)
#
#
#         self.batch_size=batch_size
#
#
#
#
#     def forward(self, x):
#
#         x,h = self.gru0(x,None)
#         x,h=self.gru1(x,None)
#
#
#         x = torch.sigmoid(x)
#         return x


class RnnGenerateNet(nn.Module):

    def __init__(self, num_features=9, gru_hidden_size=64, batch_size=4096):
        super(RnnGenerateNet, self).__init__()

        self.num_features = num_features
        self.gru_hidden_size = gru_hidden_size
        self.num_directions = 2
        self.seq_len = 16
        self.gru_out_dim = self.seq_len * self.gru_hidden_size * self.num_directions

        self.linear_hidden_size=[120,120]
        self.gru = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=1,
            hidden_size=self.gru_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )

        self.gru_out_dim = self.seq_len*self.gru_hidden_size*self.num_directions
        self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)
        self.linear0 = torch.nn.Linear(
            self.gru_out_dim,
            self.linear_hidden_size[0])
        self.bn1 = torch.nn.BatchNorm1d(self.linear_hidden_size[1])

        self.batch_size = batch_size

        self.gru_in_dim = self.seq_len* self.num_features
        self.linear0_ = torch.nn.Linear(
            self.linear_hidden_size[1],
            self.gru_in_dim)
        self.bn0_ = torch.nn.BatchNorm1d(self.gru_in_dim)

        self.bn1_ = torch.nn.BatchNorm1d(self.seq_len)
        self.linear1_ = torch.nn.Linear(
            self.num_features,
            self.num_features)
        self.bn2_ = torch.nn.BatchNorm1d(self.seq_len)

        self.gru_ = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=1,
            hidden_size=self.num_features,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        x, _ = self.gru(x, None)
        x = x.contiguous().view(
            self.batch_size,
            self.gru_out_dim)
        x = self.bn0(x)
        x = torch.tanh(self.linear0(x))
        x = self.bn1(x)

        x = torch.tanh(self.bn0_(self.linear0_(x)))


        x = x.contiguous().view(
            self.batch_size,
            self.seq_len,
            self.num_features)
        # print(x.size(), "X SIZE")
        x, hn = self.gru_(x, None)
        x = self.bn1_(x)
        # print(x.size(),"X SIZE")
        out = self.bn2_(self.linear1_(x))
        melody = torch.sigmoid(out)

        return melody
