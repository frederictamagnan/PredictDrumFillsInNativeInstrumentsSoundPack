
import torch.nn.functional as F
import torch.nn as nn
from torch import sigmoid
class DnnNet(nn.Module):


    def __init__(self,deepdata):
        super(DnnNet, self).__init__()

        self.input_deep_shape=deepdata.shape[1]

        # self.fc1 = nn.Linear(self.input_deep_shape,1)
        # self.fc4 = nn.Linear(2,2)
        # # self.fc5=nn.Linear(2,2)
        self.output = nn.Linear(self.input_deep_shape, 1)
        self.activation = sigmoid

    def forward(self,X_d):


        # print(X_d.size(),"XD CONCATNEDATED")
        # X_d= F.relu(self.fc1(X_d))
        # X_d = F.relu(self.fc2(X_d))
        # X_d = F.relu(self.fc3(X_d))
        # X_d = F.relu(self.fc4(X_d))
        # X_d = F.relu(self.fc5(X_d))
        out = self.activation(self.output(X_d))
        return out