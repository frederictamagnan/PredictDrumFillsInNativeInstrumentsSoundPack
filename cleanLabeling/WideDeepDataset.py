from utils import x_encoding
from torch.utils.data import Dataset
import numpy as np
import os
from MultiDrumOneHotEncoding import MultiDrumOneHotEncoding
import pypianoroll as ppr


FILTER=0
temp_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'

class WideDeepDataset(Dataset):
    """Dataset class for the wide and deep model
    Parameters:
    --------
    data: RawToOrganized Object
    """


    def __init__(self, data_raw,dataset=FILTER, subdir=None):

        if dataset is not None:
            data_raw=self.filter_by_(dataset,data_raw)


        data={}
        data['X'],indexes=x_encoding(data_raw['X']) #replace data['X'] by encoding with VAE
        print("data X",data['X'].shape)
        data['y_fake']=np.zeros((data['X'].shape[0],1))
        labels_deep=['y_velocity','y_offbeat','X']
        labels_wide = ['y_used','y_genre']

        self.indexes=indexes
        self.subdir=[subdir[i] for i in indexes]
        self.data_raw=data_raw
        for key in data_raw.keys():
            if key not in ["X"]:
                data[key]=data_raw[key][indexes]
        list_data_deep=[]
        for key in labels_deep:
            list_data_deep.append(data[key])

        self.X_deep=np.concatenate(list_data_deep,axis=1)

        list_data_wide=[]
        for key in labels_wide:
            list_data_wide.append(data[key])
        wide=np.concatenate(list_data_wide,axis=1)
        self.X_wide=wide.reshape((wide.shape[0],wide.shape[1]))

        Y=data['y_fills'][:,1,0]
        self.Y=Y

        print(self.X_wide.shape,"WIDE SHAPE")
        print(self.X_deep.shape,"DEEP SHAPE")
        print(self.Y.shape,"Y SHAPE",self.Y.sum(),"nb fills")





    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y  = self.Y[idx]

        return xw, xd, y

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
        #todo
        pass
    def mix_dataset(self,ratio1,ratio2):
        #todo
        pass

    def listen (self,idx):
        decoder = MultiDrumOneHotEncoding()
        print(self.subdir[idx])
        track= self.data_raw['X'][self.indexes][idx]
        print(track.shape,"track shape")
        print(track.sum(),"sum")
        track_decoded=decoder.encoded_pianoroll_to_multitrack(track)
        ppr.write(track_decoded,temp_path+'song.mid')









