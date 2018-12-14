
from torch.utils.data import Dataset
class WideDeepDataset(Dataset):
    """Dataset class for the wide and deep model
    Parameters:
    --------
    data: RawToOrganized Object
    """
    def __init__(self, data):

        self.X_wide = data.wide
        self.X_deep = data.deep
        self.Y = data.labels

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y  = self.Y[idx]

        return xw, xd, y

    def __len__(self):
        return len(self.Y)