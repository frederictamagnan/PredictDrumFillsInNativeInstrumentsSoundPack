import torch
from utils import product
import torch.nn.functional as F
import torch.nn as nn

class WideDeepNet(nn.Module):


    def __init__(self,widedata,deepdata):
        super(WideDeepNet, self).__init__()

        self.input_deep_shape=product(deepdata.shape[1:])

        self.fc1 = nn.Linear(self.input_deep_shape, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 10)
        self.output = nn.Linear(10+1, 1)
        self.activation = F.sigmoid

    def forward(self, X_w,X_d):
        # Max pooling over a (2, 2) window
        X_d=X_d.view(-1,self.input_deep_shape)
        print(X_d.size(),"XD CONCATNEDATED")
        X_d= F.relu(self.fc1(X_d))
        X_d = F.relu(self.fc2(X_d))
        X_d = F.relu(self.fc3(X_d))
        X_d = F.relu(self.fc4(X_d))

        wd = torch.cat([X_d, X_w.float()], 1)
        out = self.activation(self.output(wd))
        return out