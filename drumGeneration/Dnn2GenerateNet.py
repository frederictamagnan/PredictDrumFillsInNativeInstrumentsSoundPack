import torch.nn as nn
import torch.nn.functional as F

import torch
class Dnn2GenerateNet(nn.Module):

    def __init__(self,num_features=9,gru_hidden_size=64,batch_size=4096):
        super(Dnn2GenerateNet, self).__init__()

        self.num_features=num_features
        self.gru_hidden_size=gru_hidden_size
        self.num_directions=1
        self.seq_len=16
        self.batch_size=batch_size

        self.fc0 = nn.Linear(self.seq_len*self.num_features, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 144)




    def forward(self, x):

        x = x.contiguous().view(
            self.batch_size,self.seq_len*self.num_features)

        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = torch.sigmoid(x)
        x = x.contiguous().view(
            self.batch_size,
            self.seq_len,
            self.num_features)
        return x
