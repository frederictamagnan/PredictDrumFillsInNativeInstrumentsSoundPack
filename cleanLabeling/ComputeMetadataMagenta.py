import os
from pypianoroll import Multitrack,Track
import numpy as np
from Metadata import Metadata




class ComputeMetadataLPD:

    def __init__(self, filepath_dataset):

        self.filepath_dataset = filepath_dataset



        # if not (os.path.isfile(self.filepath_dataset + "labels.npz")):
        #     np.savez(self.filepath_dataset+"labels.npz",empty=np.empty([2, 2]))




    def macro_iteration(self,whole_dataset=False):


        # ITERATE OVER THE TAG LISTS

        for i, filename in enumerate(os.listdir(self.filepath_dataset)):

            for npz in os.listdir(self.filepath_dataset+'/'+filename):
                if 'label' not in npz and 'metadata' not in npz and 'metrics' not in npz:

                    self.process_npz_file(self.filepath_dataset+'/'+filename,npz)





    def process_npz_file(self,path,npz):
        # print(path,npz)
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
        metadata=Metadata(track)
        metadata.save_metadata(path+"/",npz[:-4])



        #enregistre metadata
        #predit label
        #enregistrelabel
        # print("done")


if __name__=='__main__':



    PATH = "/home/ftamagna/Documents/_AcademiaSinica/dataset/magentaDrums/"


    data=ComputeMetadataLPD(PATH)

    data.macro_iteration(whole_dataset=True)