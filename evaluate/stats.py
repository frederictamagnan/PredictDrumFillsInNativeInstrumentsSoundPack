PATH = '/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'


PATH_TAGS = [
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Blues.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Country.id',



'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Electronic.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Folk.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Jazz.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Latin.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Metal.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_New-Age.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Pop.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Punk.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rap.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Reggae.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_RnB.id',
    '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
'/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_World.id',

    ]
import os
from random import randint
from pypianoroll import


def random_file(filepath_dataset=PATH,path_tags=PATH_TAGS):
        count=0
        all=[]

        # ITERATE OVER THE TAG LISTS

        for tag_i, tag in enumerate(path_tags):

            print('>>' + tag[29:-3])
            with open(tag, 'r') as f:
                # ITERATE OVER THE FOLDER LISTS

                for i, file in enumerate(f):
                    # (str(f))
                    #                 print('load files..{}/{}'.format(i + 1, number_files[tag_i]), end="\r")
                    file = file.rstrip()
                    middle = '/'.join(file[2:5]) + '/'
                    p = filepath_dataset + middle + file

                    for npz in os.listdir(p):
                        if 'metadata' not in npz and 'label' not in npz:
                            count+=1




        return count

print(random_file())

print(sum([19, 448, 352, 33, 127, 70, 112, 29, 1086, 12, 89, 53, 264, 1668, 10]))