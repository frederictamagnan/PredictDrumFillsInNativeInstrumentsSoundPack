import numpy as np
import os
from MultiDrumOneHotEncoding import MultiDrumOneHotEncoding
from pypianoroll import Multitrack,Track
import pypianoroll as ppr
import time
from pathlib import Path
TIME = time.strftime("%Y%m%d_%H%M%S")
import sys
ROOTDIR="/home/ftamagna/Documents/_AcademiaSinica/dataset/oddgrooves/ODDGROOVES_FILL_PACK/OddGrooves Fill Pack General MIDI"
# ROOTDIR="/home/ftamagna/Documents/_AcademiaSinica/dataset//NI_Drum_Studio_Midi_encoded/MIDI Files"


newdir="/home/ftamagna/Documents/_AcademiaSinica/dataset/TotalFills/"

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



    def __init__(self):


        self.rootdir=ROOTDIR
        self.drum_encoding = MultiDrumOneHotEncoding()

        self.list_genre=["Pop","Funk","Jazz","Hard","Metal","Blues & Country","Blues Rock","Ballad","Indie Rock","Indie Disco","Punk Rock"]






    def folders_to_clean_numpy(self):
        list_filepath=[]
        filldifferent1=0
        filldifferent2=0
        beat_resolution=True

        X=np.zeros((1,96,9))
        y_fills=np.zeros((1,2,1))
        y_genre=np.zeros((1,12,1))
        y_bpm=np.zeros((1,1))

        y_dataset=np.zeros((1,2,1))
        i=0
        error=0

        for filepath in Path(ROOTDIR).glob('./**/*'):
            if filepath.is_file():
                print(filepath)
                try:
                    i+=1
                    filepath=str(filepath)
                    multi= Multitrack(filepath)
                    multi.binarize()

                    list_multi=self.crop(multi)

                    while(len(list_multi)>0):

                        multi=list_multi[0]

                        #Process the fills label
                        if  "Fill" in filepath or "OddGrooves" in filepath:
                            label_fills = np.array([[0], [1]])
                        else:
                            label_fills = np.array([[1], [0]])

                        #process the genre label
                        if "OddGrooves" in filepath:
                            label_genre = np.zeros((12, 1))
                            label_genre[11] = 1
                        else:
                            label_genre=self.return_label_genre(filepath)

                        #process the BPM and type

                        if "OddGrooves" in filepath:
                            index_bpm=filepath.index("BPM")
                            index_middle=filepath.index("-")
                            bpm=int(filepath[index_middle+3:index_bpm-1])
                            label_bpm=np.full(shape=(1,1),fill_value=bpm)
                            label_dataset=np.array([[0],[1]])

                        else:
                            list_=['Groove','Fill']
                            offset=[8,6]
                            index_=int(label_fills[1,0])
                            print(index_)
                            string_=list_[index_]
                            offset_=offset[index_]
                            index_bpm = filepath.index("BPM")
                            print(string_)
                            index_middle = filepath.index(string_)
                            bpm=int(filepath[index_middle + offset_:index_bpm])
                            label_bpm=np.full(shape=(1,1),fill_value=bpm)

                            label_dataset = np.array([[1], [0]])



                        encoded = self.drum_encoding.multitrack_to_encoded_pianoroll(multi)
                        #compute some extra data on encoded





                        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])


                        label_genre=label_genre.reshape(1,label_genre.shape[0],label_genre.shape[1])
                        label_fills=label_fills.reshape(1,label_fills.shape[0],label_fills.shape[1])
                        label_dataset=label_dataset.reshape(1,label_dataset.shape[0],label_dataset.shape[1])

                        X=np.concatenate((X,encoded))
                        y_fills=np.concatenate((y_fills,label_fills))
                        y_genre=np.concatenate((y_genre,label_genre))
                        y_bpm=np.concatenate((y_bpm,label_bpm))
                        y_dataset=np.concatenate((y_dataset,label_dataset))


                        logger.debug("new track encoded stacked")
                        list_multi.pop(0)
                        list_filepath.append(filepath)




                except IndexError:
                    logger.debug("--FAILED TO LOAD THE MIDI FILE because index error ERROR#"+str(error))
                    error+=1

        X=X[1:,:,:]
        y_genre=y_genre[1:,:]
        y_fills = y_fills[1:,:]
        y_bpm=y_bpm[1:,:]
        y_dataset=y_dataset[1:,:]


        np.savez(newdir+'dataset_odd_'+TIME+'.npz',X=X,y_genre=y_genre,y_fills=y_fills,y_bpm=y_bpm,y_dataset=y_dataset)
        print("errors count : ",error)
        print(i,"nb loop")
        print("fill different less",filldifferent2,"mre",filldifferent1)
        print(beat_resolution)

    def load_dataset(self,filepath):

        data=np.load(filepath)
        return data['X'],data['y']



    def crop(self,multi):

        list_multi=[]
        logger.debug("try to crop or pad and return a list")
        length = multi.tracks[0].pianoroll.shape[0]
        height = multi.tracks[0].pianoroll.shape[1]

        if length<96:
            padding = np.zeros((96 - length, height))
            print(multi.tracks[0].pianoroll.shape,padding.shape)
            a=np.concatenate((padding, multi.tracks[0].pianoroll))
            print(a)
            multi.tracks[0].pianoroll =a
            list_multi.append(multi)
            return list_multi
        else:
            i=0
            while (i + 1) * 96 <= length:
                new_multi=multi.copy()
                new_multi.tracks[0].pianoroll=multi.tracks[0].pianoroll[i*96:(i+1)*96]
                list_multi.append(new_multi)
                i+=1

        return list_multi



    def return_label_genre(self,subdir):

        label_genre=np.zeros((12,1))

        for i,elt in enumerate(self.list_genre):

            if elt in subdir:
                label_genre[i,0]=1
                print("RETURN LABEL")
                return label_genre

        raise "Error finding genre"


    def count_use_cymbals(self,multi):








datasetBuilder=DatasetBuilder()
datasetBuilder.folders_to_clean_numpy()



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





# statistics(newdir+'dataset_NI_encoded_-24.npz')