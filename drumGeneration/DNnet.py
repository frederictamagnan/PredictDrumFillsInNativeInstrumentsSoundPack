import torch.nn as nn
import torch.nn.functional as F
import torch
class DNnet(nn.Module):
    def __init__(self,batch_size=256):
        super(DNet, self).__init__()

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x


