import numpy as np
from sklearn.externals import joblib
import os
class Labelling:

    def __init__(self,filepath_dataset,filepath_tags):


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
                        if '_metadata_training' in npz:
                            self.label(p,npz)


    def label(self,path,npz):

        data=dict(np.load(path+'/'+npz))
        vm=data['velocity_metadata']
        list_vm=[]
        index_vm=[22,23,24,25,19]
        for i in index_vm:
            m=((vm[:, i] > vm[:, i].max() * 0.75)*1).reshape(1,-1)
            print(m.shape,"m shape")
            list_vm.append(m)

        vm_ = np.concatenate(list_vm,axis=0)
        print(vm_.shape)
        y=np.sum(vm_,axis=0)>=3
        print(y.shape,"y shape")
        # y=self.clf.predict(X)
        np.savez(path+'/' + npz.replace('_metadata_training.npz','') + '_label_explicit.npz', label=y)



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

