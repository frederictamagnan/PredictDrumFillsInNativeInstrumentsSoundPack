import torch
from DNnet import DNnet
import numpy as np
from utils import random_file_genre
from pypianoroll import Track,Multitrack
from DrumsDataset_vae_genre import DrumsDataset
from DrumReducerExpander import DrumReducerExpander
import pypianoroll as ppr
from utils import numpy_drums_save_to_midi
from torch.autograd import Variable
from utils import tensor_to_numpy
import torch.utils.data
from VaeEncoderDecoder_808_9 import VaeEncoderDecoder
from models.vae_rnn import *
from decimal import *

class Generator:

    def __init__(self,model_path,model_name,dataset,temp_filepath):
        self.temp_filepath=temp_filepath
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')

        self.dataset=dataset
        self.model_path=model_path
        self.model_name=model_name
        self.load_model()


    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params,"PARAMETERS")

    def load_model(self):
        self.model = DNnet()
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path + self.model_name, map_location="cuda" if self.use_cuda else "cpu"))



    def generate(self,n,save=True):
        self.load_model()
        decoder = DrumReducerExpander()
        decoderVAE = VaeEncoderDecoder()

        data=dict(np.load(self.dataset))
        array_vae=data['vae'][:n]
        genre=data['genre'][:n]
        # array_vae=array_vae.reshape((array_vae.shape[0],3,32,2))
        array_vae = array_vae.reshape((array_vae.shape[0], 2, 32, 2))

        X_dataset = DrumsDataset(numpy_array=array_vae, genre=genre, inference=True, use_cuda=self.use_cuda)
        X_loader = torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                               shuffle=False, drop_last=False)
        with torch.no_grad():
            for i, (x, g) in enumerate(X_loader):
                x = Variable(x).float()
                print(x)
                y_pred = self.model(x, g)

        y_pred = tensor_to_numpy(y_pred)
        y_pred = y_pred.reshape((y_pred.shape[0], 32, 2))
        # y_pred[:,:,1]=1
        print(y_pred[0])
        drums_reduced = decoderVAE.decode_to_reduced_drums(y_pred)

        print(drums_reduced.shape, "shape drums red")

        l = drums_reduced.shape[0]
        # threshold=self.search_treshold(array=drums_reduced,number_notes_min=l*5,number_notes_max=l*15)
        drums_reduced = drums_reduced > 0.75  # 0.76159416
        # drums_reduced=drums_reduced>0.76159416
        # print(drums_reduced.sum(),"number Notes")
        drums_reduced = decoder.decode_808(drums_reduced)
        drums = decoder.decode(drums_reduced)
        # print(drums.shape,"drumshape")
        # print(X_old.shape,"X OLD")
        X_old=data['track_array'][:n]
        print(X_old.shape,"X old shape")
        X_old=X_old.reshape(X_old.shape[0],16*2,9)
        X_old=decoder.decode_808(X_old)
        X_old=decoder.decode(X_old)
        # X_old_r=X_old.reshape(X_old.shape[0],3,96,128)
        X_old_r = X_old.reshape(X_old.shape[0], 2, 96, 128)

        # X_new=np.concatenate((X_old_r[:,0,:,:],drums,X_old_r[:,2,:,:],),axis=1)
        X_new = np.concatenate((X_old_r[:, 0, :, :], X_old_r[:, 0, :, :], drums, drums, drums, drums), axis=1)

        # print(X_new.shape,"x new shape")
        if save == True:
            for i in range(len(X_old)):
                # numpy_drums_save_to_midi(X_old_r[i].reshape(288,128),self.temp_filepath,list_filepath[i][1]+"_original")
                numpy_drums_save_to_midi(X_old_r[i].reshape(192, 128), self.temp_filepath, str(i)+"_original")

                numpy_drums_save_to_midi(X_new[i], self.temp_filepath,str(i)+"_new")
                print(X_new[i].shape)


    # def generate(self,n,save=True):
    #     self.load_model()
    #     list_filepath=[(str(i),str(i)) for i in range(n)]
    #     decoder=DrumReducerExpander()
    #     decoderVAE=VaeEncoderDecoder()
    #     dataset_vae_genre=dict(np.load(self.dataset_vae_path))
    #     dataset_fills=dict(np.load(self.dataset_fills_path))
    #
    #     leng=dataset_vae_genre['vae'].shape[0]
    #     index=np.random.randint(0,leng,n)
    #     array_vae=dataset_vae_genre['vae'][index]
    #     genre=dataset_vae_genre['genre'][index]
    #     X_reduced=dataset_fills['fills'][index]
    #     X_reduced=X_reduced.reshape((n,-1,9))
    #     X_expanded=decoder.decode(X_reduced)
    #     X_expanded=decoder.decode_808(X_expanded)
    #     print(X_expanded.shape)
    #     # X_old=X_expanded.reshape((n,3,96,128))
    #     X_old=X_expanded.reshape((n,2,96,128))
    #
    #
    #
    #     X_dataset=DrumsDataset(numpy_array=array_vae,genre=genre,inference=True,use_cuda=self.use_cuda)
    #     X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
    #                                                          shuffle=False,drop_last=False)
    #     with torch.no_grad():
    #         for i, (x,g) in enumerate(X_loader):
    #             x = Variable(x).float()
    #             y_pred = self.model(x,g)
    #
    #     y_pred=tensor_to_numpy(y_pred)
    #     y_pred=y_pred.reshape((y_pred.shape[0],32,2))
    #     drums_reduced=decoderVAE.decode_to_reduced_drums(y_pred)
    #     drums_reduced=decoder.decode_808(drums_reduced)
    #     # print(drums_reduced.shape,"shape drums red")
    #     l=drums_reduced.shape[0]
    #     # threshold=self.search_treshold(array=drums_reduced,number_notes_min=l*8,number_notes_max=l*20)
    #     drums_reduced=drums_reduced>0.75  #0.76159416
    #     # drums_reduced=drums_reduced>0.76159416
    #     # print(drums_reduced.sum(),"number Notes")
    #
    #     drums=decoder.decode(drums_reduced)
    #     # print(drums.shape,"drumshape")
    #     # print(X_old.shape,"X OLD")
    #
    #     # X_old_r=X_old.reshape(X_old.shape[0],3,96,128)
    #     # X_new=np.concatenate((X_old_r[:,0,:,:],drums,X_old_r[:,2,:,:],),axis=1)
    #     X_old_r = X_old.reshape(X_old.shape[0], 2, 96, 128)
    #     X_new = np.concatenate((X_old_r[:, 0, :, :],X_old_r[:, 0, :, :],drums, drums), axis=1)
    #     # print(X_new.shape,"x new shape")
    #     if save==True:
    #         for i in range(len(X_old)):
    #             numpy_drums_save_to_midi(X_old_r[i].reshape(192,128),self.temp_filepath,list_filepath[i][1]+"_original")
    #             numpy_drums_save_to_midi(X_new[i], self.temp_filepath, list_filepath[i][1] + "_new")
    #
    #
    #     # print(X_new[0])


    def search_treshold(self,array,number_notes_min=10,number_notes_max=15):
        getcontext().prec=1000
        max = Decimal(10)
        min = Decimal(0)

        number_notes = 3000



        i = 0
        while not (number_notes >= number_notes_min and number_notes <= number_notes_max):
            t = Decimal((min + max) / 2)
            # print(t)
            # print(threshold)
            array_b = array > t
            print(t)
            number_notes = array_b.sum()
            # print(number_notes,"NOTES")
            # print(number_notes)

            if number_notes_max < number_notes:
                min = t

            else:
                max = t

            i = i + 1
            # print(i,"COUNTER")
            if i>100:
                break
        print(i,"FINAL ITERATION")
        return t


if __name__=='__main__':


    server=False

    if server:
        model_path = '/home/ftamagnan/PredictDrumFillsInNativeInstrumentsSoundPack/models/'
        model_name = 'generation_model.pt'

    else:
        model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
        model_name = 'vae_generation.pt'


        dataset='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/ununbest.npz'
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'




    g=Generator(model_path=model_path,model_name=model_name,dataset=dataset,temp_filepath=temp_filepath)
    g.count_parameters()
    # g.generate(10,save=False)
    g.generate(500, save=True)


