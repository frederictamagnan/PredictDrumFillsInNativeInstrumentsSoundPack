import torch.utils.data as data
import numpy as np
import torch
from torch.autograd import Variable

class DrumsDataset(data.Dataset):

    def __init__(self, numpy_array,use_cuda=False,inference=False):
        self.inference=inference
        X_previous=numpy_array[:,0,:,:]
        X_next=numpy_array[:,2,:,:]

        self.X=np.concatenate((X_previous,X_next),axis=1)
        self.X=self.X.reshape((-1,2,96,9))
        if not inference:
            self.y=numpy_array[:,1,:,:]
            self.y=self.y.reshape((-1,96*9))
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

class EmbeddingsDataset(data.Dataset):

    def __init__(self,embeddings):

        mu=embeddings[:,:,0]
        sigma=embeddings[:,:,1]

        self.X=self._sample_latent(mu,sigma)

    def __getitem__(self,idx):

        return self.X[idx]

    def __len__(self):
        return self.X.shape[0]

    def _sample_latent(self, mu, sigma):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu=torch.from_numpy(mu).float()
        sigma=torch.from_numpy(sigma).float()
        std_z = torch.from_numpy(
            np.random.normal(0, 1, size=sigma.size())
        ).float()

        return mu + sigma * Variable(std_z, requires_grad=False).float()
