import torch
from SketchRnnDataset import SketchRnnDataset
from SketchRnnNet import SketchRnnNet
from SketchRnnNet import SketchEncoder
from SketchRnnNet import SketchDecoder
from sklearn.externals import joblib
from tools.Metadata import Metadata
from scipy.stats import chisquare
import numpy as np
import pandas as pd
from torch.autograd import Variable
from utils import tensor_to_numpy
from utils import numpy_drums_save_to_midi
import torch.utils.data
import scipy.stats as stats
from DrumReducerExpander import DrumReducerExpander
import time
import torch

class GeneratorSketchRnn:

    def __init__(self,model_path,model_name,dataset_path,tags_path,temp_filepath,indices_path,tag):
        self.temp_filepath=temp_filepath
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # if self.use_cuda:
        #     print('run on GPU')
        # else:
        #     print('run on CPU')

        self.dataset_path = dataset_path
        self.tags_path = tags_path
        self.model_path = model_path
        self.model_name = model_name
        self.linear_hidden_size=[64,32]
        self.gru_hidden_size=64
        self.indices_path=indices_path
        self.tag=tag
        max=1000
        factor=1
        offset=0
        torch.manual_seed(0)
        np.random.seed(0)
        self.load_model(batch_size=max*factor)

        # dataset = dict(np.load(self.dataset_path))
        # indices = dict(np.load(self.indices_path))
        # dataset = dataset['track_array']
        #
        # #train validation creation from indices
        # self.train=dataset[indices['train']]
        # self.validation=dataset[indices['validation']]
        #
        # #Filter
        # # self.train = self.filter(self.train)
        # # self.validation = self.filter(self.validation)
        #
        #
        # #multiplication
        # self.train = np.tile(self.train[offset:max+offset],(factor,1,1,1))
        # self.validation =np.tile( self.validation[offset:max+offset],(factor,1,1,1))
        #
        # #shuffle
        # np.random.shuffle(self.train)
        # np.random.shuffle(self.validation)

        #RAW
        path = '/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/RawFiles/NI_raw.npz'
        data = np.load(path)

        data = dict(data)['X'][:1000]
        encoder=DrumReducerExpander()
        data=encoder.encode(data)
        data=encoder.encode_808(data)
        self.train = data.reshape((data.shape[0],1,data.shape[1],data.shape[2]))
        self.validation=np.zeros(self.train.shape)

        # print(self.train.shape,self.validation.shape,'FILTER SHAPE')
        # self.train = dataset[indices['train']][:max]
        # self.validation = dataset[indices['validation']][:max]
        self.count_parameters()



        self.clf = joblib.load('./tools/warmness.pkl')
        self.scaler = joblib.load('./tools/scaler.pkl')


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
        # print("GOOD")
        self.model.load_state_dict(torch.load(self.model_path + self.model_name, map_location="cuda" if self.use_cuda else "cpu"))


    def generate(self,save=False):

        for i_dataset,dataset in enumerate([self.train,self.validation]):
            X_dataset=SketchRnnDataset(numpy_array=dataset,inference=True,use_cuda=self.use_cuda)
            # print("LOAD DATASET GOOD")
            X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=len(X_dataset),
                                                                 shuffle=False,drop_last=True,worker_init_fn=lambda x : 0)
            with torch.no_grad():
                for i, (x) in enumerate(X_loader):
                    x = Variable(x).float()
                    y_pred = self.model(x)
                    y_pred_cat = (y_pred >0.15)

            self.y_gen=tensor_to_numpy(y_pred_cat).astype(int)
            # self.y_or=(self.train[:,1,:,:]>0)*1
            self.y_or=np.zeros(self.y_gen.shape)
            self.x_or = (self.train[:, 0, :, :]>0)*1
            print(self.y_gen.shape,self.y_or.shape,self.x_or.shape,"GEN")
            break
        if save:
            encoder=DrumReducerExpander()
            new = np.concatenate((self.x_or,self.x_or,self.x_or,self.y_gen,self.x_or,self.x_or,self.x_or,self.y_gen), axis=1)
            new_dec = encoder.decode(new)
            new_dec = encoder.decode_808(new_dec)

            for i in range(len(new_dec)):
                numpy_drums_save_to_midi(new_dec[i], self.temp_filepath +self.tag+ "_new",str(i)+'_')



    def testchi2(self):

        # print(self.y_gen.shape,"shape")
        dist_x_or = self.x_or.reshape((-1, 9))
        dist_y_gen=self.y_gen.reshape((-1,9))
        dist_y_or=self.y_or.reshape((-1,9))
        print(dist_x_or.sum(),dist_y_gen.sum(),dist_y_or.sum(),"SUM")
        dist_y_gen = self.y_gen.reshape((-1, 9))[:,4:8]
        dist_y_or = self.y_or.reshape((-1, 9))[:,4:8]
        # dist_y_gen=(dist_y_gen-dist_x_or>0)*1
        # dist_y_or=(dist_y_or-dist_x_or>0)*1


        list_gen=[]
        list_or=[]
        list_x_or=[]
        for i in range(dist_y_gen.shape[1]):
            list_gen.append(dist_y_gen[:, i].sum())
            list_or.append(dist_y_or[:, i].sum())
            list_x_or.append(dist_x_or[:, i].sum())


        list_gen=np.asarray(list_gen)
        list_or=np.asarray(list_or)
        list_x_or = np.asarray(list_x_or)

        # print(dist_y_gen, "PRINT DIST")
        # list_gen=list_gen/dist_y_gen.sum()
        # list_or=list_or/dist_y_or.sum()
        # list_x_or=list_x_or/dist_x_or.sum()
        print(list_gen,list_or,list_x_or)

        chisq,pvalue=chisquare(list_gen,f_exp=list_x_or)

        # print(dist_y_gen,"PRINT DIST")
        # chi_squared_stat = (np.power(list_gen- list_or,2) / list_or).sum()
        # pvalue = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
        #                              df=8)
        # statt,pvalue=stats.wilcoxon(list_gen,list_or)

        # print(pvalue)
        arr=np.asarray([list_gen,list_x_or])
        chi2,pvalue,dof,expected=stats.chi2_contingency(arr)

        return pvalue


    def warmness(self):


        dec=DrumReducerExpander()
        predictions=[]
        for i,elt in enumerate([self.y_or,self.y_gen]):
            elt_=dec.decode(elt)
            elt_=dec.decode_808(elt_)
            data=Metadata(elt_).metadata
            data['vae_embeddings'] = data['vae_embeddings'][:, 0:32]
            list_label = ['vae_embeddings','count','drums_pitches_used']
            list_x = []
            for label in list_label:
                list_x.append(data[label])
            X = np.concatenate(list_x, axis=1)
            X_std = self.scaler.transform(X)
            # y = (self.clf.predict_proba(X_std) > 0.7) * 1
            y = self.clf.predict_proba(X_std)
            y = y[:, 1]
            predictions.append(y)

        # warmness=np.power((predictions[0]-predictions[1]),2).sum()/len(predictions[0])
        warmness=np.absolute((predictions[0]-predictions[1])).sum()/len(predictions[0])

        return warmness

    def filter(self,dataset,n=5):
        print(dataset.shape)
        sumi=dataset[:,0,:,:]
        sumi = sumi.reshape((sumi.shape[0], 16 * 9))
        sumi = np.sum(sumi, axis=1)
        dataset = dataset[np.where(sumi > n)]
        return dataset

























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
row_warmness=[]
row_testchi2=[]
for i_name,name in enumerate(['Clustering','Supervised','Diff']):

    t_warmness=[]
    t_chi2=[]
    for beta in [0.01,0.1,1,100,250]:
        dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtracted'+name+'.npz'
        indices_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/indices/indices_sketchrnn_'+name+'_'+str(beta)+'.pt.npz'
        model_name='sketchrnn_'+name+'_'+str(beta)+'_cleaned.pt'
        # temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'+folder[i_name]+'/'+str(beta)+'/'
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
        g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath,indices_path=indices_path,tag="_"+str(beta)+"_"+name+"_")
        g.count_parameters()
        # g.generate(10,save=False)
        g.generate()
        t_warmness.append(g.warmness())
        t_chi2.append(g.testchi2())

    t_warmness=tuple(t_warmness)
    t_chi2=tuple(t_chi2)

    row_warmness.append(t_warmness)
    row_testchi2.append(t_chi2)


df_w = pd.DataFrame(row_warmness, columns = ['0.01','0.1','1','100','250'], index=['Clustering','Supervised','Diff'])
df_t = pd.DataFrame(row_testchi2, columns = ['0.01','0.1','1','100','250'], index=['Clustering','Supervised','Diff'])

print("Original Dataframe" , df_w, sep='\n')
print("Original Dataframe" , df_t, sep='\n')

# g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
#         g.count_parameters()
#         # g.generate(10,save=False)
#         g.generate(10, save=True)