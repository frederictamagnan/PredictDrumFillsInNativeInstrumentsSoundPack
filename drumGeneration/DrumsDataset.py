import torch.utils.data as data


class DrumsDatasetataset(data.Dataset):

    def __init__(self, numpy_array):
        X_previous=numpy_array[:,0,96,9]
        X_next=numpy_array[:,2,96,9]

        self.X=np.concatenate((X_previous,X_next),axis=1)
        self.y=numpy_array[:,1,96,9]

    def __getitem__(self, index):
        X = self.data[index, :, 0:24, :]
        y = self.data[index, :, 24:96, :]
        #         X.reshape(np.prod(X.shape))
        y = y.reshape(np.prod(y.shape))
        # print(y.shape,"lol")
        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)

        return X, y

    def __len__(self):
        return len(self.data)
