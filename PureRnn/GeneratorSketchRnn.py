import torch
from SketchRnnDataset import SketchRnnDataset
from SketchRnnNet import SketchRnnNet
from SketchRnnNet import SketchEncoder
from SketchRnnNet import SketchDecoder
from pypianoroll import Track,Multitrack
import numpy as np
from utils import random_file
from DrumReducerExpander import DrumReducerExpander

from utils import numpy_drums_save_to_midi
from torch.autograd import Variable
from utils import tensor_to_numpy
import torch.utils.data
class GeneratorSketchRnn:

    def __init__(self,model_path,model_name,dataset_path,tags_path,temp_filepath):
        self.temp_filepath=temp_filepath
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')

        self.dataset_path = dataset_path
        self.tags_path = tags_path
        self.model_path = model_path
        self.model_name = model_name
        self.linear_hidden_size=[64,32]
        self.gru_hidden_size=64



        self.load_model(10)
        self.count_parameters()


    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params,"beta vae parameters")

    def load_model(self,batch_size):
        encoder = SketchEncoder(batch_size=batch_size,linear_hidden_size=self.linear_hidden_size, gru_hidden_size=self.gru_hidden_size).to(
            self.device)
        decoder = SketchDecoder(batch_size=batch_size,linear_hidden_size=self.linear_hidden_size).to(self.device)
        # self.model = SketchRnnNet(encoder, decoder).to(self.device)
        self.model=decoder
        self.model.eval()

        print("GOOD")
        self.model.load_state_dict(torch.load(self.model_path + self.model_name, map_location="cuda" if self.use_cuda else "cpu"))


    def generate(self,n,save=True):
        self.load_model(batch_size=n)
        encoder=DrumReducerExpander()
        X=np.zeros((1,192,128))
        list_filepath=[]
        i=0
        while i<n:
            rf=random_file()
            list_filepath.append(rf)
            multi=Multitrack(rf[0]+rf[1])
            multi.binarize()
            piano=multi.tracks[0].pianoroll
            length=len(piano)
            mid=int(length//96//2)
            x=piano[mid*96:(mid+2)*96]
            x=x.reshape((1,x.shape[0],x.shape[1]))
            # print(x.shape)
            try:
                # print(X.shape)
                X=np.concatenate((X,x))
                i+=1
            except:
                print("error")

        X_old=X[1:]
            # print(x.shape)
        X=encoder.encode(X_old)
        X=encoder.encode_808(X)
        X=X.reshape((X.shape[0],2,16,9))
        X_dataset=SketchRnnDataset(numpy_array=X,inference=True,use_cuda=self.use_cuda)
        print("LOAD DATASET GOOD")
        X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                                             shuffle=False,drop_last=True)
        with torch.no_grad():
            for i, (x) in enumerate(X_loader):
                x = Variable(x).float()
                y_pred = self.model(x)
                y_pred_cat = (y_pred > 0.25)
        # y_pred = tensor_to_numpy(y_pred).astype(float)
        # y_pred_cat=np.zeros(y_pred.shape)


        # t=np.array([0.3,0.20,0.3,0.3,0.20,0.20,0.20,0.2,0.30])

        # t = np.array([0.3, 0.20, 0.3, 0.3, 0.20, 0.20, 0.20, 0.2, 0.30])
        # for i in range(len(t)):
        #     y_pred_cat[:,:,i]=(y_pred[:,:,i]>t[i])*1


        y_pred_cat=tensor_to_numpy(y_pred_cat).astype(int)
        y=y_pred_cat.reshape((n,16, 9))
        print(y[0])
        # print(X[1, 0, :, :],"value")
        new = np.concatenate((X[:, 0, :, :],X[:, 0, :, :],X[:, 0, :, :], y,X[:, 0, :, :],X[:, 0, :, :],X[:, 0, :, :], y),axis=1)
        print(new.shape)
        # print(new[0].sum())
        new_dec=encoder.decode(new)
        new_dec=encoder.decode_808(new_dec)

        # print(new_dec.shape)

        if save==True:
            for i in range(len(X)):
                numpy_drums_save_to_midi(X_old[i].reshape(192,128),self.temp_filepath,list_filepath[i][1]+"_original")
                numpy_drums_save_to_midi(new_dec[i], self.temp_filepath, list_filepath[i][1] + "_new")

        new2 = np.concatenate((X[:, 0, :, :], y), axis=1)
        # np.save(temp_filepath+'all',new2)
        print(y.shape)



    def generate_from(self,array,tag,save=True,th=0.15):
        n=len(array)
        self.load_model(batch_size=n)
        encoder = DrumReducerExpander()
        X=encoder.encode(array)
        X=encoder.encode_808(X)
        X=X.reshape((X.shape[0],1,16,9))
        X_dataset=SketchRnnDataset(numpy_array=X,inference=True,use_cuda=self.use_cuda)
        X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                                             shuffle=False,drop_last=True)
        with torch.no_grad():
            for i, (x) in enumerate(X_loader):
                x = Variable(x).float()
                y_pred = self.model(x)
                y_pred_cat = (y_pred >th)
        y_pred_cat=tensor_to_numpy(y_pred_cat).astype(int)
        y=y_pred_cat.reshape((n,16, 9))
        new = np.concatenate((X[:, 0, :, :],X[:, 0, :, :],X[:, 0, :, :], y,X[:, 0, :, :],X[:, 0, :, :],X[:, 0, :, :], y),axis=1)
        new_dec=encoder.decode(new)
        new_dec=encoder.decode_808(new_dec)

        if save==True:
            for i in range(len(X)):
                numpy_drums_save_to_midi(new_dec[i], self.temp_filepath, "sample_"+str(i) +tag)

    def generate_from_magenta(self, array, tag, save=True, th=0.15):
        n = len(array)
        self.load_model(batch_size=n)
        encoder = DrumReducerExpander()
        X=array
        X_dataset = SketchRnnDataset(numpy_array=X, inference=True, use_cuda=self.use_cuda)
        X_loader = torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                               shuffle=False, drop_last=True)
        with torch.no_grad():
            for i, (x) in enumerate(X_loader):
                x = Variable(x).float()
                y_pred = self.model(x)
                y_pred_cat = (y_pred > th)
        y_pred_cat = tensor_to_numpy(y_pred_cat).astype(int)
        y = y_pred_cat.reshape((n, 16, 9))
        y[:,:,1]=0
        new = np.concatenate(
            (X[:, 0, :, :], X[:, 0, :, :], X[:, 0, :, :], y, X[:, 0, :, :], X[:, 0, :, :], X[:, 0, :, :], y), axis=1)

        # np.savez(self.temp_filepath+tag,track_array=y)
        new_dec = encoder.decode(new)

        new_dec = encoder.decode_808(new_dec)

        # old = np.concatenate(
        #     (array[:, 0, :, :], array[:, 0, :, :], array[:, 0, :, :], array[:, 1, :, :], array[:, 0, :, :],
        #      array[:, 0, :, :], array[:, 0, :, :], array[:, 1, :, :]), axis=1)

        # old_dec = encoder.decode(old)
        # old_dec = encoder.decode_808(old_dec)

        if save == True:
            pass
            for i in range(len(X)):
                numpy_drums_save_to_midi(new_dec[i], self.temp_filepath, "sample_" + str(i) + tag)
                # numpy_drums_save_to_midi(old_dec[i], self.temp_filepath, "sample_" + str(i) + tag+"_magenta")

    def generate_long(self,tag,array,n=10,save=True,th=0.30):
        self.load_model(batch_size=1)
        encoder = DrumReducerExpander()

        list_x=[]
        X = encoder.encode(array,no_batch=True)
        X = encoder.encode_808(X,no_batch=True)

        list_x.append(X)
        print(X.shape,"X SHAPE 0")
        for i in range(n):
            X = X.reshape((1, 1, 16, 9))
            X_dataset = SketchRnnDataset(numpy_array=X, inference=True, use_cuda=self.use_cuda)
            X_loader = torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                                   shuffle=False, drop_last=False)

            with torch.no_grad():
                for i, (x) in enumerate(X_loader):
                    x = Variable(x).float()
                    y_pred = self.model(x)
                    y_pred_cat = (y_pred >th)
            y_pred_cat=tensor_to_numpy(y_pred_cat).astype(int)
            y_pred_cat=y_pred_cat.reshape((16,9))
            list_x.append(y_pred_cat),
            X=y_pred_cat
            print(X.shape,"X SHAPE")

        new=np.concatenate(list_x,axis=0)
        print(new.shape,"NEWWWW")
        new_dec = encoder.decode(new,no_batch=True)
        new_dec = encoder.decode_808(new_dec,no_batch=True)

        if save == True:

            numpy_drums_save_to_midi(new_dec, self.temp_filepath, "sample_" +tag)












if __name__=='__main__':


    server=False

    if server:
        model_path = '/home/ftamagnan/PredictDrumFillsInNativeInstrumentsSoundPack/models/'
        model_name = 'generation_model.pt'

    else:
        model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
        model_name = 'sketchrnn_Supervised_v4.pt'

        dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'
        tags_path= ['/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id']
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'


# folder=['c','s','d']
# for i_name,name in enumerate(['Clustering','Supervised','Diff']):
#     for beta in [1,10,250]:
#
#         model_name='sketchrnn_'+name+'_'+str(beta)+'.pt'
#         temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'+folder[i_name]+'/'+str(beta)+'/'
#         g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
#         g.count_parameters()
#         # g.generate(10,save=False)
#         g.generate(10, save=True)

    import numpy as np
    np.random.seed(10)
    for i in range(10):
        g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
        g.count_parameters()
                # g.generate(10,save=False)
        g.generate(1, save=True)