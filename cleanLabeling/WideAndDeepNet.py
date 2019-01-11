import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import sigmoid
class WideAndDeepNet(nn.Module):


    def __init__(self,widedata,deepdata):
        super(WideAndDeepNet, self).__init__()

        self.input_deep_shape=deepdata.shape[1]
        self.input_wide_shape=widedata.shape[1]
        self.fc1 = nn.Linear(self.input_deep_shape,3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, 3)
        self.output = nn.Linear(3+self.input_wide_shape, 1)
        self.activation = sigmoid

    def forward(self, X_w,X_d):


        # print(X_d.size(),"XD CONCATNEDATED")
        X_d= F.relu(self.fc1(X_d))
        X_d = F.relu(self.fc2(X_d))
        X_d = F.relu(self.fc3(X_d))
        X_d = F.relu(self.fc4(X_d))

        wd = torch.cat([X_d, X_w.float()], 1)
        out = self.activation(self.output(wd))
        return out