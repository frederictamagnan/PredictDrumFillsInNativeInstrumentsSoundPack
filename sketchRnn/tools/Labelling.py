from DnnNet import DnnNet
import numpy as np
import torch
from torch.autograd import Variable
from utils import tensor_to_numpy
import os
from DnnDataset import DnnDataset
class Labelling:

    def __init__(self,filepath_model,filename_model,filepath_dataset,filepath_tags):

        fake_deep_data=np.zeros((1,32+9+1+36,1))

        self.model=DnnNet(fake_deep_data)

        self.model.load_state_dict(torch.load(filepath_model+filename_model))
        self.model.eval()
        self.filepath_dataset = filepath_dataset
        self.filepath_tags = filepath_tags

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
                        if '_metrics_training' in npz:
                            self.label(p,npz)


    def label(self,path,npz):

        data=np.load(path+'/'+npz)
        y=self.predict(data)
        np.savez(path+'/' + npz.replace('_metrics_training.npz','') + '_label.npz', label=y)



    def predict(self,data):

        dataset=DnnDataset(data,[],downsampling=False,upsampling=False,inference=True)
        inference_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset),shuffle=False)
        y_pred_total = []
        with torch.no_grad():
            for i, (X_deep) in enumerate(inference_loader):
                X_d = Variable(X_deep).float()
                y_pred = self.model(X_d)
                y_pred_cat = (y_pred >0.5).squeeze(1).float()

                y_pred_total.append(tensor_to_numpy(y_pred_cat).astype(int))


        y_pred_total = np.concatenate(y_pred_total)
        return y_pred_total







if __name__=='__main__':
    PATH = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
    PATH_TAGS = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
    ]

    lb=Labelling('./',"fillClassifier",PATH,PATH_TAGS)
    lb.macro_iteration()

