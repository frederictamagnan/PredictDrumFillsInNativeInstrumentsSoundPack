import torch.utils.data as data
import numpy as np
import torch
class DrumsDataset(data.Dataset):

    def __init__(self, numpy_array,genre,use_cuda=False,inference=False):
        self.inference=inference
        X_previous=numpy_array[:,0,:,:]
        # X_next=numpy_array[:,2,:,:]

        # self.X=np.concatenate((X_previous,X_next),axis=1)
        self.X=X_previous
        # self.X=self.X.reshape((-1,128))
        self.X=self.X.reshape((-1,64))
        if not inference:
            self.y=numpy_array[:,1,:,:]
            self.y=self.y.reshape((-1,64))
        self.use_cuda=use_cuda
        self.genre=genre.reshape((genre.shape[0],15))
        print(self.genre.shape,"GENRE SHAPE")

    def __getitem__(self, index):
        X = self.X[index]
        g=self.genre[index]
        if not self.inference:
            y = self.y[index]
        #         X.reshape(np.prod(X.shape))
        # print(y.shape,"lol")

        if self.use_cuda:
            X = torch.from_numpy(X).type(torch.cuda.FloatTensor)
            g = torch.from_numpy(g).type(torch.cuda.FloatTensor)
            if not self.inference:
                y = torch.from_numpy(y).type(torch.cuda.FloatTensor)
        else:
            X = torch.from_numpy(X).type(torch.FloatTensor)
            g = torch.from_numpy(g).type(torch.FloatTensor)
            if not self.inference:
                y = torch.from_numpy(y).type(torch.FloatTensor)



        if not self.inference:
            return X,g, y
        else:
            return X,g

    def __len__(self):
        return len(self.X)
