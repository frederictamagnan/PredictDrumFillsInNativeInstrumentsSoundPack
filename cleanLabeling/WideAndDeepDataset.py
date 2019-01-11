
from torch.utils.data import Dataset
import numpy as np
import os
from DrumReducerExpander import DrumReducerExpander
import pypianoroll as ppr


FILTER=0
temp_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'

class WideAndDeepDataset(Dataset):
    """Dataset class for the wide and deep model
    Parameters:
    --------
    data: RawToOrganized Object
    """


    def __init__(self, data, list_filepath,dataset=None,upsampling=True):

        # if dataset is not None:
        #     self.data=self.filter_by_(dataset,data)

        if list_filepath is None:
            raise "arg list filepath required"

        self.data=data
        labels_deep=['vae_embeddings','offbeat_notes','velocity_metrics']
        # labels_deep=['vae_embeddings']
        labels_wide=['genre','drums_pitches_used']


        list_data_deep=[]
        for key in labels_deep:
            list_data_deep.append(data[key])
            print(data[key].shape,"check shape")
        self.X_deep=np.concatenate(list_data_deep,axis=1)


        list_data_wide = []
        for key in labels_wide:
            list_data_wide.append(data[key])
            print(data[key].shape,"check shape")
        self.X_wide=np.concatenate(list_data_wide,axis=1)


        Y=data['fills'][:,1,0]
        self.Y=Y
        self.list_filepath=list_filepath
        if upsampling:
            self.upsampling()

        self.shuffle()

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y  = self.Y[idx]

        return xw,xd, y

    def __len__(self):
        return len(self.Y)


    def filter_by_(self,dataset_number,data_raw):
        y_dataset = data_raw['y_dataset']
        print(y_dataset.shape)
        indexes_dataset=np.argwhere(y_dataset[:,1,0]==dataset_number).reshape(-1)
        print(indexes_dataset.shape)
        # indexes_dataset = y_dataset[y_dataset == dataset_number,1,0]

        for key in data_raw.keys():

            data_raw[key]=data_raw[key][indexes_dataset]

        return data_raw

    def upsampling(self):
        print(len(self.X_deep),"before upsampling")
        indexes_0=np.argwhere(self.Y==0).reshape(-1)
        indexes_1 = np.argwhere(self.Y == 1).reshape(-1)
        indexes_0_reduced=np.random.choice(indexes_0,size=len(indexes_1),replace=False)


        # print(indexes_1_reduced[:3])

        self.X_deep=np.concatenate((self.X_deep[indexes_0_reduced],self.X_deep[indexes_1]))
        self.X_wide=np.concatenate((self.X_wide[indexes_0_reduced],self.X_wide[indexes_1]))

        self.list_filepath=[self.list_filepath[i] for i in indexes_0_reduced]+[self.list_filepath[i] for i in indexes_1]

        self.Y=np.concatenate((self.Y[indexes_0_reduced],self.Y[indexes_1]))
        print(len(self.X_deep),"after upsampling")

    def shuffle(self):
        indexes_shuffled=np.random.choice(len(self.X_deep),size=len(self.X_deep),replace=False)
        self.X_deep=self.X_deep[indexes_shuffled]
        self.X_wide = self.X_wide[indexes_shuffled]

        self.list_filepath=[self.list_filepath[i] for i in indexes_shuffled]


    def listen (self,idx):
        decoder = DrumReducerExpander()
        print(self.list_filepath[idx])
        track= self.data['X'][idx]
        print(track.shape,"track shape")
        print(track.sum(),"sum")
        track=track.reshape(1,track.shape[0],track.shape[1])
        track_decoded=decoder.decode(track)[0]
        track_ppr=Track(track_decoded)

        ppr.write(track_ppr,temp_path+'song.mid')

