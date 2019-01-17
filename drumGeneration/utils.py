PATH = '/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'
PATH_TAGS = [
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Blues.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Country.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Electronic.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Folk.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Jazz.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Latin.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Metal.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_New-Age.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Pop.id', # 8
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Punk.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Rap.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Reggae.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_RnB.id',
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_Rock.id', # 13
        '/home/herman/lpd/id_lists/tagtraum/tagtraum_World.id',
    ]

PATH_TAGS_ROCK = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
    ]
import os
from random import randint

from pypianoroll import Track,Multitrack
import pypianoroll as ppr

def tensor_to_numpy(array):

    return array.cpu().data.numpy()



def random_file(filepath_dataset=PATH,path_tags=PATH_TAGS_ROCK):

        all=[]

        # ITERATE OVER THE TAG LISTS

        for tag_i, tag in enumerate(path_tags):

            # print('>>' + tag[29:-3])
            with open(tag, 'r') as f:
                # ITERATE OVER THE FOLDER LISTS

                for i, file in enumerate(f):
                    # (str(f))
                    #                 print('load files..{}/{}'.format(i + 1, number_files[tag_i]), end="\r")
                    file = file.rstrip()
                    middle = '/'.join(file[2:5]) + '/'
                    p = filepath_dataset + middle + file

                    for npz in os.listdir(p):
                        if 'metrics' not in npz and 'label' not in npz:
                            all.append((p+'/',npz))


        pick=all[randint(0,len(all))]
        return pick

def numpy_drums_save_to_midi(array,filepath,filename):

    track=Track(array,is_drum=True)
    multi=Multitrack(tracks=[track])
    ppr.write(multi,filepath+filename)


def tensor_to_numpy(array):

    return array.cpu().data.numpy()