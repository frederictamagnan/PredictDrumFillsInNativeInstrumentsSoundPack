from torch.utils.data import Dataset



class DrumsDataset(Dataset):
    """Drums dataset."""

    def __init__(self,X):

        self.X=X


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):

        return self.X[idx,:,:],idx




