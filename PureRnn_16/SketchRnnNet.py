import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable






class SketchDecoder(nn.Module):

    def __init__(self,batch_size=256,num_features=9,seq_len=16,linear_hidden_size=[64,32]):
        super(SketchDecoder, self).__init__()
        self.num_features = num_features
        self.gru_2_hidden=128
        self.seq_len = seq_len
        self.linear_hidden_size = linear_hidden_size
        self.batch_size=batch_size
        self.gru = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=2,
            hidden_size=self.gru_2_hidden,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.bn1 = torch.nn.BatchNorm1d(self.seq_len)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.linear1 = torch.nn.Linear(
            self.gru_2_hidden,
            num_features)
        # self.linear2=torch.nn.Linear(64,self.num_features)

    def forward(self, x):

        x,hz=self.gru(x)
        # print(x.size(),"x after GRU")
        x=self.bn1(x)
        # x = x.view(x.size()[0],self.gru_2_hidden*self.seq_len)
        x=self.linear1(x)


        # x=F.relu(self.linear2(x))
        # x = x.contiguous().view(x.size()[0], 48,9)
        x = self.bn2(x)
        x = torch.sigmoid(x)

        # print(x.size(),"x final")
        return x





