import os
from pypianoroll import Multitrack,Track
import numpy as np
from Metrics import Metrics




class ComputeMetricsLPD:

    def __init__(self, filepath_dataset, filepath_tags):

        self.filepath_dataset = filepath_dataset
        self.filepath_tags = filepath_tags


        # if not (os.path.isfile(self.filepath_dataset + "labels.npz")):
        #     np.savez(self.filepath_dataset+"labels.npz",empty=np.empty([2, 2]))




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
                        if 'label' not in npz and 'metrics' not in npz:
                            self.process_npz_file(p,npz)





    def process_npz_file(self,path,npz):
        print(path,npz)
        multi=Multitrack(path+"/"+npz)
        track=multi.tracks[0].pianoroll
        # print(track.shape)
        if track.shape[0]==0:
            return 0
        if track.shape[0]%96!=0:
            to_complete_len=96-track.shape[0]%96
            to_complete=np.zeros((to_complete_len,128))
            track=np.concatenate((track,to_complete))
        track=track.reshape((track.shape[0]//96,96,128))
        metrics=Metrics(track)
        metrics.save_metrics(path+"/",npz[:-4])



        #enregistre metrics
        #predit label
        #enregistrelabel
        # print("done")


if __name__=='__main__':



    PATH = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
    PATH_TAGS = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
    ]

    data=ComputeMetricsLPD(PATH,PATH_TAGS)

    data.macro_iteration()