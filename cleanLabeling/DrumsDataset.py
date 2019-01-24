from torch.utils.data import Dataset
import torch
import numpy as np
from torch.autograd import Variable


class DrumsDataset(Dataset):
    """Drums dataset."""

    def __init__(self,X):

        self.X=X


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):

        return self.X[idx,:,:],idx


class EmbeddingsDataset(Dataset):

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



