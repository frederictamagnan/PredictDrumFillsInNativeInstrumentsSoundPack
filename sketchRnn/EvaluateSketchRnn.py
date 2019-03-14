import torch
from SketchRnnDataset import SketchRnnDataset
from SketchRnnNet import SketchRnnNet
from SketchRnnNet import SketchEncoder
from SketchRnnNet import SketchDecoder


import numpy as np

from torch.autograd import Variable
from utils.utils2 import tensor_to_numpy
import torch.utils.data
class GeneratorSketchRnn:

    def __init__(self,model_path,model_name,dataset_path,tags_path,temp_filepath,indices_path):
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
        self.indices_path=indices_path

        self.load_model(batch_size=700)

        dataset = dict(np.load(self.dataset_path))
        indices = dict(np.load(self.indices_path))
        dataset = dataset['track_array']

        self.train = dataset[indices['train']][:700]
        self.validation = dataset[indices['validation']][:700]
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
        self.model = SketchRnnNet(encoder, decoder).to(self.device)
        self.model.eval()
        print("GOOD")
        self.model.load_state_dict(torch.load(self.model_path + self.model_name, map_location="cuda" if self.use_cuda else "cpu"))


    def generate(self,n,save=True):


        for i_dataset,dataset in enumerate([self.train,self.validation]):
            X_dataset=SketchRnnDataset(numpy_array=dataset,inference=True,use_cuda=self.use_cuda)
            print("LOAD DATASET GOOD")
            X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                                                 shuffle=False,drop_last=True)
            with torch.no_grad():
                for i, (x) in enumerate(X_loader):
                    x = Variable(x).float()
                    y_pred = self.model(x)
                    y_pred_cat = (y_pred >0.15)

            y_pred_cat=tensor_to_numpy(y_pred_cat).astype(int)
            y=self.train[:,1,:,:]

            
            print(y_pred_cat.shape)


















if __name__=='__main__':


    server=False

    if server:
        model_path = '/home/ftamagnan/PredictDrumFillsInNativeInstrumentsSoundPack/models/'
        model_name = 'generation_model.pt'

    else:
        model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
        model_name = 'sketchrnn.pt'

        dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'
        tags_path= ['/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id']
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
        indices_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/indices/'


folder=['c','s','d']
for i_name,name in enumerate(['Clustering','Supervised','Diff']):
    for beta in [0.01,10,250]:
        dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtracted'+name+'.npz'
        indices_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/indices/indices_sketchrnn_'+name+'_'+str(beta)+'.pt.npz'
        model_name='sketchrnn_'+name+'_'+str(beta)+'.pt'
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'+folder[i_name]+'/'+str(beta)+'/'
        g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath,indices_path=indices_path)
        g.count_parameters()
        # g.generate(10,save=False)
        g.generate(10, save=True)


# g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
#         g.count_parameters()
#         # g.generate(10,save=False)
#         g.generate(10, save=True)