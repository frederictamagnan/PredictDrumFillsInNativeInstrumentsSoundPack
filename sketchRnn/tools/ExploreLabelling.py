
import numpy as np
import os
class ExploreLabelling:

    def __init__(self,filepath_dataset,filepath_tags):
        self.filepath_dataset=filepath_dataset
        self.filepath_tags=filepath_tags




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
                            if 'label' in npz:
                                self.explore(p,npz)

    def explore(self,p,npz):
        current = dict(np.load(p +'/'+ npz))['label']
        print(current,p)



if __name__=='__main__':
    PATH = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
    PATH_TAGS = [
        '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
    ]

    ex=ExploreLabelling(PATH,PATH_TAGS)
    ex.macro_iteration()

