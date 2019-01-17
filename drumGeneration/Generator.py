import torch
from CNNNet import CNNNet
import numpy as np
from utils import random_file
from pypianoroll import Track,Multitrack
from DrumsDataset import DrumsDataset
from DrumReducerExpander import DrumReducerExpander
import pypianoroll as ppr
from utils import numpy_drums_save_to_midi
from torch.autograd import Variable
from utils import tensor_to_numpy
import torch.utils.data
class Generator:

    def __init__(self,model_path,model_name,dataset_path,tags_path,temp_filepath):
        self.temp_filepath=temp_filepath
        self.use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')

        self.load_model(256)
        self.dataset_path=dataset_path
        self.tags_path=tags_path


    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params)

    def load_model(self,batch_size):
        self.model = CNNNet(batch_size=batch_size)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path + model_name, map_location="cuda" if self.use_cuda else "cpu"))


    def generate(self,n):
        self.load_model(batch_size=n)
        encoder=DrumReducerExpander()
        X=np.zeros((1,288,128))
        list_filepath=[]
        i=0
        while i<n:
            rf=random_file()
            list_filepath.append(rf)
            multi=Multitrack(rf[0]+rf[1])
            piano=multi.tracks[0].pianoroll
            length=len(piano)
            mid=int(length//96//2)
            x=piano[mid*96:(mid+3)*96]
            x=x.reshape((1,x.shape[0],x.shape[1]))
            print(x.shape)
            X=np.concatenate((X,x))
            i+=1

        X_old=X[1:]
            # print(x.shape)
        X=encoder.encode(X_old)
        X=X.reshape((X.shape[0],3,96,9))
        X_dataset=DrumsDataset(numpy_array=X,inference=True,use_cuda=self.use_cuda)
        X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                                             shuffle=False,drop_last=False)
        with torch.no_grad():
            for i, (x) in enumerate(X_loader):
                x = Variable(x).float()
                y_pred = self.model(x)
                y_pred_cat = (y_pred >0.5)
                print(y_pred[0])

        y_pred_cat=tensor_to_numpy(y_pred_cat).astype(int)
        y=y_pred_cat.reshape((n,96, 9))
        print(y.shape)
        print(X[:, 0, :, :].shape)
        new = np.concatenate((X[:, 0, :, :], y,X[:, 2, :, :]),axis=1)
        print(new.shape)

        new_dec=encoder.decode(new)

        print(new_dec.shape)


        for i in range(len(X)):
            numpy_drums_save_to_midi(X_old[i].reshape(288,128),self.temp_filepath,list_filepath[i][1]+"_original")
            numpy_drums_save_to_midi(new_dec[i], self.temp_filepath, list_filepath[i][1] + "_new")




        print(y.shape)















if __name__=='__main__':


    server=False

    if server:
        model_path = '/home/ftamagnan/PredictDrumFillsInNativeInstrumentsSoundPack/models/'
        model_name = 'generation_model.pt'

    else:
        model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
        model_name = 'generation_model.pt'

        dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'
        tags_path= ['/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id']
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'




    g=Generator(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
    g.count_parameters()
    # g.generate(10)


