import numpy as np
import os
from pypianoroll import Multitrack,Track
import time
from pathlib import Path
import json
import sys
ROOTDIR2="/home/ftamagna/Documents/_AcademiaSinica/dataset/oddgrooves/ODDGROOVES_FILL_PACK/OddGrooves Fill Pack General MIDI"
ROOTDIR="/home/ftamagna/Documents/_AcademiaSinica/dataset//NI_Drum_Studio_Midi_encoded/MIDI Files"


newdir="/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/RawFiles/"

import logging

from logging.handlers import RotatingFileHandler

# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger()
# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(logging.DEBUG)

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
# création d'un handler qui va rediriger une écriture du log vers
# un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
# on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
# créé précédement et on ajoute ce handler au logger
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# création d'un second handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)



class DatasetBuilder:



    def __init__(self,rootdir):


        self.rootdir=rootdir




    def folders_to_clean_numpy(self):
        TIME = time.strftime("%Y%m%d_%H%M%S")

        list_filepath=[]

        X=np.zeros((1,96,128))

        i=0
        error=0

        for filepath in Path(self.rootdir).glob('./**/*'):
            if filepath.is_file():
                print(filepath)
                try:
                    i+=1
                    filepath=str(filepath)
                    multitrack= Multitrack(filepath)
                    drum_pianoroll=multitrack.tracks[0].pianoroll

                    list_multi=self.crop(drum_pianoroll)

                    while(len(list_multi)>0):
                        x=list_multi[0]
                        x=x.reshape([1]+list(x.shape))
                        X=np.concatenate((X,x))
                        list_filepath.append(filepath)
                        list_multi.pop(0)

                except IndexError:
                    logger.debug("--FAILED TO LOAD THE MIDI FILE because index error ERROR#"+str(error))
                    error+=1

            # if i > 2:
            #     break


        np.savez(newdir+'raw_concatenation_'+TIME+'.npz',X=X)

        with open(newdir + 'data'+TIME+'.json', 'w') as outfile:
            json.dump(list_filepath, outfile)


        print("errors count : ",error)
        print(i,"nb loop")

    def load_dataset(self,filepath):

        data=np.load(filepath)
        return data['X'],data['y']



    def crop(self, multi):
        list_multi = []
        logger.debug("try to crop or pad and return a list")
        length = multi.shape[0]
        height = multi.shape[1]

        if length<96:
            padding = np.zeros((96 - length, height))
            multi=np.concatenate((padding, multi))
            list_multi.append(multi)
            return list_multi
        else:
            i=0
            while (i + 1) * 96 <= length:
                new_multi=multi[i*96:(i+1)*96]
                list_multi.append(new_multi)
                i+=1
                print("LONGER")

        return list_multi


def statistics(filepath):
    data = np.load(filepath)
    X = data['X']
    y_genre = data['y_genre']
    y_fills = data['y_fills']
    y_bpm = data['y_bpm']
    # y_dataset = data['y_dataset']

    print(X.shape,"X shape")
    print(y_genre.shape,"y_genre shape")
    print(y_bpm.shape,"y_bpm shape")
    print(y_fills.shape,"y_fills shape")
    print(y_bpm.shape, "y_bpm shape")
    # print(y_dataset.shape,"y_dataset")
    print(y_fills[:,1,0].sum()," count fills")



if __name__ == '__main__':

    # datasetBuilder=DatasetBuilder(ROOTDIR)
    # datasetBuilder.folders_to_clean_numpy()
    datasetBuilder = DatasetBuilder(ROOTDIR2)
    datasetBuilder.folders_to_clean_numpy()



# statistics(newdir+'dataset_NI_encoded_-24.npz')