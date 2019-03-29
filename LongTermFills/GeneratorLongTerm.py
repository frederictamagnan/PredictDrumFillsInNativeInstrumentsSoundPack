import torch
from LongTermNet import Encoder,DecoderFills,LongTermNet
from LongTermNetDataset import LongTermNetDataset
from pypianoroll import Track,Multitrack
import numpy as np
from utils import random_file
from DrumReducerExpander import DrumReducerExpander

from utils import numpy_drums_save_to_midi
from torch.autograd import Variable
from utils import tensor_to_numpy
import torch.utils.data as Data
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
        self.num_directions=2
        self.seq_len=2
        self.gru_embedding_hidden_size=32


        self.load_model(10)
        self.count_parameters()


    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params,"beta vae parameters")

    def load_model(self,batch_size):
        self.encoderFills = Encoder(num_features=9, gru_hidden_size=self.gru_hidden_size,
                               gru_hidden_size_2=self.gru_hidden_size, seq_len=self.seq_len,
                               num_directions=self.num_directions, linear_hidden_size=self.linear_hidden_size).to(
            self.device)
        self.encoderRegular = Encoder(num_features=9, gru_hidden_size=self.gru_hidden_size,
                                 gru_hidden_size_2=self.gru_hidden_size, seq_len=self.seq_len,
                                 num_directions=self.num_directions, linear_hidden_size=self.linear_hidden_size).to(
            self.device)
        self.decoderFills = DecoderFills(linear_hidden_size=self.linear_hidden_size,
                                    gru_embedding_hidden_size=self.gru_embedding_hidden_size).to(self.device)
        self.model = LongTermNet(self.encoderFills, self.decoderFills, self.encoderRegular).to(self.device)

        self.model.eval()
        print("GOOD")
        self.model.load_state_dict(torch.load(self.model_path + self.model_name, map_location="cuda" if self.use_cuda else "cpu"))



    def generate_fills(self,n,save=True):
        sample=np.random.randn(n,2,32)
        sample=torch.from_numpy(sample).type(torch.FloatTensor)
        sample_dataset=Data.TensorDataset(sample)
        train_loader = Data.DataLoader(
            dataset=sample_dataset,
            batch_size=n,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )
        for batch_i, data in enumerate(train_loader):
            with torch.no_grad():
                # print(data[0].size())
                drums=self.decoderFills(data[0][:,0,:], data[0][:,1,:])

        print(drums.shape)
        drums=tensor_to_numpy(drums)
        print(drums[0])
        drums=(drums>0.26)*1


        enc=DrumReducerExpander()
        drums_dec=enc.decode(drums)
        drums_dec=enc.decode_808(drums_dec)


        if save==True:
            for i in range(len(drums)):
                numpy_drums_save_to_midi(drums_dec[i],self.temp_filepath,str(i))


    def generate_fills_properly(self,n,save=True):

        offset=20
        dataset=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtractedFour_cleaned_v2.npz')
        dataset=dict(dataset)['track_array'][offset:offset+n]*1
        print(dataset.shape)
        regular = torch.from_numpy(dataset).type(torch.FloatTensor)
        regular_dataset = Data.TensorDataset(regular)
        regular_loader = Data.DataLoader(
            dataset=regular,
            batch_size=n,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )
        for batch_i, data in enumerate(regular_loader):
            with torch.no_grad():
                # print(data[0].size())
                # print(type(data),data.shape)
                z_input = self.encoderRegular(data[:, 0:2, :,:].reshape((n,32,9)))

        z_input=self.model._sample_latent(z_input)[0]
        z_input = tensor_to_numpy(z_input).reshape((n,1,32))
        z_output = np.random.randn(n, 1, 32)

        z_concat=np.concatenate((z_input,z_output),axis=1)


        sample = torch.from_numpy(z_concat).type(torch.FloatTensor)
        sample_dataset = Data.TensorDataset(sample)
        train_loader = Data.DataLoader(
            dataset=sample_dataset,
            batch_size=n,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )
        for batch_i, data in enumerate(train_loader):
            with torch.no_grad():
                # print(data[0].size())
                drums = self.decoderFills(data[0][:, 0, :], data[0][:, 1, :])

        print(drums.shape)
        drums = tensor_to_numpy(drums)
        print(drums[0])
        drums = (drums > 0.27) * 1

        enc = DrumReducerExpander()
        drums_dec = enc.decode(drums)
        drums_dec = enc.decode_808(drums_dec)

        if save==True:
            for i in range(len(drums)):
                numpy_drums_save_to_midi(drums_dec[i],self.temp_filepath,str(i))



















if __name__=='__main__':


    server=False

    if server:
        model_path = '/home/ftamagnan/PredictDrumFillsInNativeInstrumentsSoundPack/models/'
        model_name = 'generation_model.pt'

    else:
        model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
        model_name = 'longterm_1st.pt'

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


g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
g.count_parameters()
        # g.generate(10,save=False)
# g.generate(10, save=True)
# g.generate_fills(n=10,save=True)
g.generate_fills_properly(n=10)