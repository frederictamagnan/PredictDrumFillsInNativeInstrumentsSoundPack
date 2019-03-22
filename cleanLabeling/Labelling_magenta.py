import numpy as np
from sklearn.externals import joblib
import os
class Labelling:

    def __init__(self,filepath_model,filename_model,filepath_dataset):


        self.filepath_dataset = filepath_dataset

        self.clf=joblib.load(filepath_model+filename_model)
        self.scaler=joblib.load(filepath_model+'scaler.pkl')

    def macro_iteration(self):


        # ITERATE OVER THE TAG LISTS

        for i, filename in enumerate(os.listdir(self.filepath_dataset)):

            for npz in os.listdir(self.filepath_dataset+'/'+filename):


                if '_metadata_training' in npz:
                    self.label(self.filepath_dataset+'/'+filename,npz)


    def label(self,path,npz):

        data=dict(np.load(path+'/'+npz))
        # print(npz,"NPZ")
        data['vae_embeddings'] = data['vae_embeddings'][:, 0:32]
        list_label=['vae_embeddings','offbeat_notes','velocity_metadata','drums_pitches_used']
        list_x=[]
        for label in list_label:
            list_x.append(data[label])
        X = np.concatenate(list_x, axis=1)
        X_std=self.scaler.transform(X)
        y=(self.clf.predict_proba(X_std)>0.95)*1
        y=y[:,1]
        # print("number of fills",y.sum())
        # y=self.clf.predict(X)
        np.savez(path+'/' + npz.replace('_metadata_training.npz','') + '_label.npz', label=y)



    def predict(self,data):
        pass

        # dataset=DnnDataset(data,[],downsampling=False,upsampling=False,inference=True)
        # inference_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset),shuffle=False)
        # y_pred_total = []
        # with torch.no_grad():
        #     for i, (X_deep) in enumerate(inference_loader):
        #         X_d = Variable(X_deep).float()
        #         y_pred = self.model(X_d)
        #         y_pred_cat = (y_pred >0.5).squeeze(1).float()
        #
        #         y_pred_total.append(tensor_to_numpy(y_pred_cat).astype(int))
        #
        #
        # y_pred_total = np.concatenate(y_pred_total)


        # return y_pred_total







if __name__=='__main__':
    PATH = "/home/ftamagna/Documents/_AcademiaSinica/dataset/magentaDrums/"


    lb=Labelling('./models/',"clf_fills.pkl",PATH)
    lb.macro_iteration()

