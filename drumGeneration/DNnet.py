import torch.nn as nn
import torch.nn.functional as F
import torch
class DNnet(nn.Module):
    def __init__(self):
        super(DNnet, self).__init__()

        # self.fc1 = nn.Linear(128+15, 64)
        self.fc1 = nn.Linear(64+16, 128)
        #
        #
        self.fc2 = nn.Linear(128,64 )
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)







    def forward(self, x,g):

        x = torch.cat([x, g.float()], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))

        return x


