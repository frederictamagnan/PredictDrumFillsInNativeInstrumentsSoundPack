import os
from pypianoroll import Multitrack,Track
import numpy as np
from Metrics import Metrics




class ComputeMetricsLPD:

    def __init__(self, filepath_dataset, filepath_tags, logger):

        self.filepath_dataset = filepath_dataset
        self.filepath_tags = filepath_tags
        self.logger = logger


        # if not (os.path.isfile(self.filepath_dataset + "labels.npz")):
        #     np.savez(self.filepath_dataset+"labels.npz",empty=np.empty([2, 2]))




    def macro_iteration(self):


        # ITERATE OVER THE TAG LISTS

        for tag_i, tag in enumerate(self.filepath_tags):

            if tag_i == 0:
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
                            self.process_npz_file(p,npz)




    def process_npz_file(self,path,npz):
        print(path,npz)
        multi=Multitrack(path+"/"+npz)
        track=multi.tracks[0].pianoroll
        # print(track.shape)
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
        print("done")


if __name__=='__main__':




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



    data=ComputeMetricsLPD(PATH,PATH_TAGS,logger)

    data.macro_iteration()