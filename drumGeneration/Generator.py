import torch
from CNNNet import CNNNet
import numpy as np
from pypianoroll import Track,Multitrack

class Generator:

    def __init__(self,model_path,model_name,dataset_path,tags_path):

        self.use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')

        self.model = CNNNet()
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path + model_name,map_location="cuda" if self.use_cuda else "cpu"))
        self.dataset_path=dataset_path
        self.tags_path=tags_path


    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params)



    def generate(self,n):

        for i in range(n):

            pass






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




    g=Generator(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path)


