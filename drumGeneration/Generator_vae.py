from DNnet import DNnet
from utils import random_file_genre
from DrumsDataset_vae_genre import DrumsDataset
from DrumReducerExpander import DrumReducerExpander
from utils import numpy_drums_save_to_midi
from utils import tensor_to_numpy
import torch.utils.data
from VaeEncoderDecoder_808_9 import VaeEncoderDecoder
from models.vae_rnn import *
from decimal import *

class Generator:

    def __init__(self,model_path,model_name,dataset_path,tags_path,temp_filepath):
        self.temp_filepath=temp_filepath
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')

        self.dataset_path=dataset_path
        self.tags_path=tags_path
        self.model_path=model_path
        self.model_name=model_name
        self.load_model()


    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params,"PARAMETERS_DNN")

    def load_model(self):
        self.model = DNnet()
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path + self.model_name, map_location="cuda" if self.use_cuda else "cpu"))


    def generate(self,n,save=True):
        self.load_model()
        decoder=DrumReducerExpander()
        decoderVAE=VaeEncoderDecoder()

        # array_vae=np.zeros((1,3,64))
        array_vae=np.zeros((1,2,64))

        genre=np.zeros((1,15,1))
        # X=np.zeros((1,288,128))
        X=np.zeros((1,192,128))

        list_filepath=[]
        i=0
        while i<n:
            rf=random_file_genre()
            # print(rf)
            list_filepath.append(rf)
            multi=Multitrack(rf[0]+rf[1])
            multi.binarize()

            piano=multi.tracks[0].pianoroll
            length=len(piano)
            mid=int(length//96//2)
            # x=piano[mid*96:(mid+3)*96]
            x=piano[mid*96:(mid+2)*96]

            x=x.reshape((1,x.shape[0],x.shape[1]))
            g=np.zeros((1,15,1))
            # print(rf,"RF")
            g[:,rf[2],:]=1
            try:
                metrics = dict(np.load(rf[0] + rf[1].replace('.npz', '_metadata_training.npz')))
                vae = metrics['vae_embeddings']
                # vae = vae[mid:(mid + 3)].reshape((-1, 3, 64))
                vae = vae[mid:(mid + 2)].reshape((-1, 2, 64))

                X=np.concatenate((X,x))
                genre=np.concatenate((genre,g))
                array_vae=np.concatenate((array_vae,vae))
                i+=1
                # print(i,'INDEXXXXXXXXX')
            except:
                print("error")

        X_old=X[1:]
        genre=genre[1:]
        array_vae=array_vae[1:]
        # array_vae=array_vae.reshape((array_vae.shape[0],3,32,2))
        array_vae=array_vae.reshape((array_vae.shape[0],2,32,2))

        X_dataset=DrumsDataset(numpy_array=array_vae,genre=genre,inference=True,use_cuda=self.use_cuda)
        X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                                             shuffle=False,drop_last=False)
        with torch.no_grad():
            for i, (x,g) in enumerate(X_loader):
                x = Variable(x).float()
                print(x)
                y_pred = self.model(x,g)

        y_pred=tensor_to_numpy(y_pred)
        y_pred=y_pred.reshape((y_pred.shape[0],32,2))
        # y_pred[:,:,1]=1
        # print(y_pred[0])
        drums_reduced=decoderVAE.decode_to_reduced_drums(y_pred)

        # print(drums_reduced.shape,"shape drums red")

        l=drums_reduced.shape[0]
        # threshold=self.search_treshold(array=drums_reduced,number_notes_min=l*5,number_notes_max=l*15)
        drums_reduced=drums_reduced>0.75  #0.76159416
        np.save(self.temp_filepath+"generated_with_regression",drums_reduced)

        # drums_reduced=drums_reduced>0.76159416
        # print(drums_reduced.sum(),"number Notes")
        drums_reduced=decoder.decode_808(drums_reduced)
        drums=decoder.decode(drums_reduced)
        # print(drums.shape,"drumshape")
        # print(X_old.shape,"X OLD")
        X_old_enc=decoder.encode(X_old)
        X_old_enc=decoder.encode_808(X_old_enc)
        X_old_enc=X_old_enc.reshape((-1,2*16,9))
        X_old_enc=X_old_enc[:,:16,:]
        # np.save(self.temp_filepath+"generated_with_regression_previous",X_old_enc)

        # X_old_r=X_old.reshape(X_old.shape[0],3,96,128)
        X_old_r=X_old.reshape(X_old.shape[0],2,96,128)

        # X_new=np.concatenate((X_old_r[:,0,:,:],drums,X_old_r[:,2,:,:],),axis=1)
        X_new=np.concatenate((X_old_r[:,0,:,:],X_old_r[:,0,:,:],drums,drums,drums,drums),axis=1)


        # print(X_new.shape,"x new shape")
        if save==True:
            for i in range(len(X_old)):
                # numpy_drums_save_to_midi(X_old_r[i].reshape(288,128),self.temp_filepath,list_filepath[i][1]+"_original")
                numpy_drums_save_to_midi(X_old_r[i].reshape(192,128),self.temp_filepath,list_filepath[i][1]+"_original")

                numpy_drums_save_to_midi(X_new[i], self.temp_filepath, list_filepath[i][1] + "_new")
                print(X_new[i].shape)


        # print(X_new[0])


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
            if i>1000:
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
        model_name = 'vae_generation_onelayer.pt'

        dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'
        tags_path= ['/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id']
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'




    g=Generator(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
    g.count_parameters()
    # g.generate(10,save=False)
    g.generate(10, save=True)


