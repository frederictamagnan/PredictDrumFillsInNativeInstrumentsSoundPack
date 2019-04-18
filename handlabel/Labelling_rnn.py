import numpy as np
from RnnDataset import RnnDataset
from RnnNet import RnnNet
import torch
import os
from utils import tensor_to_numpy
class Labelling:

    def __init__(self,filepath_dataset,filepath_tags):


        self.filepath_dataset = filepath_dataset
        self.filepath_tags = filepath_tags
        self.use_cuda = torch.cuda.is_available()

        device=torch.device("cuda" if self.use_cuda else "cpu")

        self.rnn = RnnNet()
        self.rnn.load_state_dict(torch.load('./../models/rnndetection.pt', map_location=device))

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
        # print(npz,"NPZ")
        raw=data['reduced_drums']
        print(raw.shape)
        X=np.stack((raw[:-2,:,:],raw[1:-1,:,:],raw[2:,:,:]))
        dataset=RnnDataset(X=X ,inference=False,use_cuda=True)
        dataloader=torch.utils.data.DataLoader(dataset=dataset,batch_size=len(dataset),shuffle=False,drop_last=False)


        for batch_i, data in enumerate(dataloader):
            with torch.no_grad():
                x= data
                y = self.rnn(x)

        y=tensor_to_numpy(y)
        y=(y>0.9)*1
        y=np.concatenate((np.zeros(1),y,np.zeros(1)))

        np.savez(path+'/' + npz.replace('_metadata_training.npz','') + '_label_rnn.npz', label=y)



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

