import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable


class RnnNet(nn.Module):


    def __init__(self,batch_size=128,num_features=9,seq_len=16*3):
        super(RnnNet, self).__init__()
        self.num_features = num_features
        self.gru_hidden=5
        self.seq_len = seq_len

        self.batch_size=batch_size
        self.gru = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=1,
            hidden_size=self.gru_hidden,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.bn1 = torch.nn.BatchNorm1d(self.seq_len)
        self.linear1 = torch.nn.Linear(
            self.gru_hidden*1*seq_len,
            1)

        # self.linear2=torch.nn.Linear(10,1)


    def forward(self, x):
        x,hz=self.gru(x)
        x=self.bn1(x)
        x=x.view((x.size()[0],-1))
        # x=F.relu(self.linear1(x))
        x = self.linear1(x)
        x = torch.sigmoid(x)

        return x


