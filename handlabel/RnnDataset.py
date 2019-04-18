import torch.utils.data as data
import numpy as np
import torch
from torch.autograd import Variable

class RnnDataset(data.Dataset):

    def __init__(self, X,y,use_cuda=False,inference=False):
        self.inference=inference
        self.X=(X>0)*1

        self.X=self.X.reshape((-1,3*16,9))

        if not inference:
            self.y=y.reshape((y.shape[0],1))
            # print(self.y[400:440])
#         print(self.X.shape)
#         print(self.y.shape)
        self.use_cuda=use_cuda

    def __getitem__(self, index):
        X = self.X[index]

        if not self.inference:
            y = self.y[index]
        #         X.reshape(np.prod(X.shape))
        # print(y.shape,"lol")

        if self.use_cuda:
            X = torch.from_numpy(X).type(torch.cuda.FloatTensor)
            if not self.inference:
                y = torch.from_numpy(y).type(torch.cuda.FloatTensor)
        else:
            X = torch.from_numpy(X).type(torch.FloatTensor)
            if not self.inference:
                y = torch.from_numpy(y).type(torch.FloatTensor)

        if not self.inference:
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X)