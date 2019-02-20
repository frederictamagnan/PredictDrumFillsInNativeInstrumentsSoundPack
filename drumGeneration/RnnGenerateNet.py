import torch.nn as nn
import torch.nn.functional as F

import torch
class RnnGenerateNet(nn.Module):

    def __init__(self,num_features=9,gru_hidden_size=64,batch_size=4096):
        super(RnnGenerateNet, self).__init__()

        self.num_features=num_features
        self.gru_hidden_size=gru_hidden_size
        self.num_directions=1
        self.seq_len=16
        self.gru_out_dim = self.seq_len * self.gru_hidden_size * self.num_directions
        #
        self.gru0 = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=3,
            hidden_size=64,
            bias=True,
            batch_first=True,
            bidirectional=False)

        self.gru1 = torch.nn.GRU(
            input_size=64,
            num_layers=6,
            hidden_size=9,
            bias=True,
            batch_first=True,
            bidirectional=False)


        self.batch_size=batch_size




    def forward(self, x):

        x,h = self.gru0(x,None)
        x,h=self.gru1(x,None)


        x = torch.sigmoid(x)
        return x
