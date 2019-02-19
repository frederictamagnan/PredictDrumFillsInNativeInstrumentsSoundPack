import torch.nn as nn
import torch.nn.functional as F

import torch
class RnnGenerateNet(nn.Module):

    def __init__(self,num_features=9,gru_hidden_size=16,batch_size=4096):
        super(RnnGenerateNet, self).__init__()

        self.num_features=num_features
        self.gru_hidden_size=gru_hidden_size
        self.num_directions=1
        self.seq_len=16
        self.gru_out_dim = self.seq_len * self.gru_hidden_size * self.num_directions

        self.gru = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=1,
            hidden_size=self.gru_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=False)
        self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)
        self.linear0 = torch.nn.Linear(
            self.gru_out_dim,
            self.gru_out_dim)
        self.gru2 = torch.nn.GRU(
            input_size=self.gru_hidden_size,
            num_layers=1,
            hidden_size=self.num_features,
            bias=True,
            batch_first=True,
            bidirectional=False)

        self.batch_size=batch_size

        self.gru_out_dim = self.seq_len * self.gru_hidden_size * self.num_directions
        #
        # self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)


    def forward(self, x):

        x,h = self.gru(x,None)
        # print(x.size(),"size after layer 1")
        x = x.contiguous().view(
            self.batch_size,
            self.gru_out_dim)
        x = self.bn0(x)
        x=F.relu(self.linear0(x))
        x = x.contiguous().view(
            self.batch_size,
            self.seq_len,
            self.gru_hidden_size)

        x,h=self.gru2(x,None)
        # x = x.contiguous().view(
        #     self.batch_size,
        #     self.gru_out_dim)
        # x=self.bn0(x)
        # print(x.size(),"size after layer 2")
        x = torch.sigmoid(x)
        return x
