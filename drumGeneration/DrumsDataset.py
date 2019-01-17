import torch.utils.data as data
import numpy as np
import torch
class DrumsDataset(data.Dataset):

    def __init__(self, numpy_array,use_cuda=False):
        X_previous=numpy_array[:,0,:,:]
        X_next=numpy_array[:,2,:,:]

        self.X=np.concatenate((X_previous,X_next),axis=1)
        self.X=self.X.reshape((-1,2,96,9))
        self.y=numpy_array[:,1,:,:]
        self.y=self.y.reshape((-1,96*9))
        self.use_cuda=use_cuda

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        #         X.reshape(np.prod(X.shape))
        # print(y.shape,"lol")

        if self.use_cuda:
            X = torch.from_numpy(X).type(torch.cuda.FloatTensor)
            y = torch.from_numpy(y).type(torch.cuda.FloatTensor)
        else:
            X = torch.from_numpy(X).type(torch.FloatTensor)
            y = torch.from_numpy(y).type(torch.FloatTensor)


        return X, y

    def __len__(self):
        return len(self.X)
