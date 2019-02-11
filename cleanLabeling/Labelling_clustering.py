import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans
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
        X=data['velocity_metadata']
        random_state = 170
        km = KMeans(n_clusters=6, random_state=random_state)
        y_pred = km.fit_predict(X)

        max=-10
        clust_max=0
        for i in range(6):
            tab_index_bar = np.argwhere(y_pred == i).reshape(-1)
            x=X[tab_index_bar][:,23:26]
            sum_=np.sum(x)
            if sum_>max:
                max=sum_
                clust_max=i

        y=y_pred[y_pred==clust_max]*1
        np.savez(path+'/' + npz.replace('_metadata_training.npz','') + '_label_clustering.npz', label=y)











if __name__=='__main__':
    PATH = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
    PATH_TAGS = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
    ]

    lb=Labelling('./models/',"clf_fills.pkl",PATH,PATH_TAGS)
    lb.macro_iteration()

