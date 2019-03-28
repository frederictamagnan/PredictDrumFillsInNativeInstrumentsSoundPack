import torch.utils.data as data
import numpy as np
import torch
from torch.autograd import Variable

class LongTermNetDataset(data.Dataset):

    def __init__(self, numpy_array,bars_input=2,bars_output=2,use_cuda=False,inference=False):

        self.inference=inference
        numpy_array=(numpy_array>0)*1
        print(numpy_array.shape, "SHAPE NUMPU")
        X_previous=numpy_array[:,0:bars_input,:,:]
        self.X=X_previous
        self.X=self.X.reshape((-1,16*bars_input,9))

        if not inference:
            self.y=numpy_array[:,bars_input:bars_input+bars_output,:,:]
            self.y=self.y.reshape((-1,16*bars_output,9))
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

