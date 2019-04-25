import torch.utils.data as data
import numpy as np
import torch
from torch.autograd import Variable
from DrumReducerExpander import DrumReducerExpander
class SketchRnnDataset(data.Dataset):

    def __init__(self, numpy_array,use_cuda=False,inference=False):
        self.inference=inference
        numpy_array=(numpy_array>0)*1

        numpy_array=numpy_array.reshape((numpy_array.shape[0],4*96,9))
        print(numpy_array.shape)
        enc=DrumReducerExpander()
        numpy_array=enc.encode_808(numpy_array)
        numpy_array=numpy_array.reshape((numpy_array.shape[0],4,16,9))

        X_previous=numpy_array[:,2,:,:]
        self.X=X_previous
        self.X=self.X.reshape((-1,16,9))

        if not inference:
            self.y=numpy_array[:,3,:,:]
            self.y=self.y.reshape((-1,16,9))
            print(self.y.shape,"Y SHAPE")
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