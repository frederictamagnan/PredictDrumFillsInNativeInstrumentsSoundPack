import numpy as np

from sklearn.cluster import KMeans
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
        print(npz,"NPZ")
        X=data['velocity_metadata']
        random_state = 170
        km = KMeans(n_clusters=6, random_state=random_state)
        y_pred = km.fit_predict(X)
        print(y_pred.shape,"y_pred_shape")
        number_notes_in_fills=4
        max=number_notes_in_fills*50/16
        clust_max=-1
        most_frequent=np.bincount(y_pred).argmax()
        for i in range(6):
            if i==most_frequent:
                pass
            else:
                tab_index_bar = np.argwhere(y_pred == i)
                tab_index_bar=tab_index_bar.reshape(-1)
                print(tab_index_bar,"tab index bar")
                print(X.shape)
                x=np.concatenate((X[tab_index_bar][:,22:26],X[tab_index_bar][:,19]),axis=1)
                print(x.shape)
                print(x,"X")
                sum_=np.sum(x)
                print(sum_,"SUM")
                if sum_>max*len(tab_index_bar):
                    max=sum_
                    clust_max=i

        if clust_max==-1:
            y=np.zeros(y_pred.shape)
        else:
            y=(y_pred==clust_max)*1

        y[y_pred==most_frequent]=-1
        print(y,"YYYY")
        print(y)
        np.savez(path+'/' + npz.replace('_metadata_training.npz','') + '_label_clustering.npz', label=y)











if __name__=='__main__':
    PATH = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
    PATH_TAGS = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
    ]

    lb=Labelling(PATH,PATH_TAGS)
    lb.macro_iteration()

