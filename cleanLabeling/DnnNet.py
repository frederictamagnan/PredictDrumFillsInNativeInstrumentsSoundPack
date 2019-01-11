
import torch.nn.functional as F
import torch.nn as nn
from torch import sigmoid
class DnnNet(nn.Module):


    def __init__(self,deepdata):
        super(DnnNet, self).__init__()

        self.input_deep_shape=deepdata.shape[1]

        self.fc1 = nn.Linear(self.input_deep_shape,20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
        self.activation = sigmoid

    def forward(self,X_d):


        # print(X_d.size(),"XD CONCATNEDATED")
        X_d= F.relu(self.fc1(X_d))
        X_d = F.relu(self.fc2(X_d))
        X_d = F.relu(self.fc3(X_d))
        out = self.activation(self.output(X_d))
        return out