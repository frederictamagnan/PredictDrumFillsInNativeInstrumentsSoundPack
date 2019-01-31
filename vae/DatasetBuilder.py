
import numpy as np
from pypianoroll import Multitrack,Track
from DrumReducerExpander import DrumReducerExpander
import os

class DatasetBuilder:

    def __init__(self,path_dataset,path_tags,save_data_path):

        self.path_dataset=path_dataset
        self.path_tags=path_tags
        self.ed=DrumReducerExpander()
        self.save_data_path=save_data_path


    def macro_iteration(self):

        # ITERATE OVER THE TAG LISTS

        for tag_i, tag in enumerate(self.path_tags):

            data=np.zeros((1,16,4))
            print('>>' + tag[29:-3])
            with open(tag, 'r') as f:
                # ITERATE OVER THE FOLDER LISTS

                for i, file in enumerate(f):
                    # (str(f))
                    #                 print('load files..{}/{}'.format(i + 1, number_files[tag_i]), end="\r")
                    self.file = file.rstrip()
                    self.middle = '/'.join(self.file[2:5]) + '/'
                    p = self.path_dataset + self.middle + self.file

                    for npz in os.listdir(p):
                        if 'metrics' not in npz and 'label' not in npz:
                            elt=self.open_track(p, npz)
                            if elt is not None:
                                print(elt.shape)

                                data=np.concatenate((data,elt))
            data=data[1:]
            np.save(self.save_data_path+tag[-7:-3],data)


    def open_track(self,p,npz):

        multi=Multitrack(p+'/'+npz)
        multib=multi.copy()
        multib.binarize()
        pb=multib.tracks[0].pianoroll.sum()
        if pb<25:
            return None
        pianoroll=multi.tracks[0].pianoroll
        nb_bar=pianoroll.shape[0]//96

        pianoroll=pianoroll[:nb_bar*96]
        pianoroll=pianoroll.reshape(nb_bar,96,128)

        pianoroll_reduced=self.ed.encode(pianoroll)
        pianoroll_reduced=self.ed.encode_808(pianoroll_reduced)
        return pianoroll_reduced














if __name__=='__main__':
    PATH = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
    PATH_TAGS = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id']

    save_data_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/trainingVAE/'

    d=DatasetBuilder(path_dataset=PATH,path_tags=PATH_TAGS,save_data_path=save_data_path)
    d.macro_iteration()