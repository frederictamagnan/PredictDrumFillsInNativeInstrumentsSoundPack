import numpy as np
from sklearn.externals import joblib
import os
class Labelling:

    def __init__(self,filepath_model,filename_model,filepath_dataset,filepath_tags):


        self.filepath_dataset = filepath_dataset
        self.filepath_tags = filepath_tags
        self.clf=joblib.load(filepath_model+filename_model)

    def macro_iteration(self):


        # ITERATE OVER THE TAG LISTS

        for tag_i, tag in enumerate(self.filepath_tags):


            print('>>' + tag[29:-3])
            with open(tag, 'r') as f:
                # ITERATE OVER THE FOLDER LISTS

                for i, file in enumerate(f):
                    # (str(f))
                    #                 print('load files..{}/{}'.format(i + 1, number_files[tag_i]), end="\r")
                    self.file = file.rstrip()
                    self.middle = '/'.join(self.file[2:5]) + '/'
                    p = self.filepath_dataset + self.middle + self.file

                    for npz in os.listdir(p):
                        if '_metadata_training' in npz:
                            self.label(p,npz)


    def label(self,path,npz):

        data=dict(np.load(path+'/'+npz))
        print(npz,"NPZ")
        data['vae_embeddings'] = data['vae_embeddings'][:, 0:32]
        list_label=['vae_embeddings','offbeat_notes','velocity_metadata','drums_pitches_used']
        list_x=[]
        for label in list_label:
            list_x.append(data[label])
        X = np.concatenate(list_x, axis=1)
        y=(self.clf.predict_proba(X)>0.7)*1
        y=y[:,1]
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
    PATH = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
    PATH_TAGS = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
    ]

    lb=Labelling('./models/',"clf_fills.pkl",PATH,PATH_TAGS)
    lb.macro_iteration()

