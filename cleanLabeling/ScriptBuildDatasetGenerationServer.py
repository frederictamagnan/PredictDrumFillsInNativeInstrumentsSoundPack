
# path_tags= [
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Blues.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Country.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Electronic.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Folk.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Jazz.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Latin.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Metal.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_New-Age.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Pop.id', # 8
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Punk.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Rap.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Reggae.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_RnB.id',
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_Rock.id', # 13
#     '/home/herman/lpd/id_lists/tagtraum/tagtraum_World.id',
# ]



import os
import numpy as np
from pypianoroll import Track, Multitrack
from DrumReducerExpander import DrumReducerExpander

def macro_iteration(filepath_dataset, filepath_tags ,max=5000 ,reduced=False ,server=True):

    enc=DrumReducerExpander()
    fills = np.zeros((1, 3 ,96, 9))
    count = 0

    # ITERATE OVER THE TAG LISTS

    for tag_i, tag in enumerate(filepath_tags):

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
                    if 'label.npz' in npz:
                        count += 1
                        fill = build_generation_dataset(p, npz)

                        if fill is not None:
                            fill = enc.encode(fill)
                            fills = np.concatenate((fills, fill))

                        if fills.shape[0 ] >max:
                            fills = fills[1:]
                            np.savez("./fills", fills=fills)
                            return 0







def build_generation_dataset(p, npz):
    label = dict(np.load(p + '/' + npz))
    label = label['label']
    #     print(label.shape)
    multi = Multitrack(p + '/' + npz.replace('_label', ''))
    track = multi.tracks[0].pianoroll
    if track.shape[0] % 96 != 0:
        to_complete_len = 96 - track.shape[0] % 96
        to_complete = np.zeros((to_complete_len, 128))
        track = np.concatenate((track, to_complete))
    track = track.reshape((track.shape[0] // 96, 96, 128))

    label_previous_next_shape = np.concatenate((label[:-2], label[1:-1], label[2:])).reshape((-1 ,3))


    mask_fills_cleaned =(label_previous_next_shape ==[0, 1, 0]).all(axis=1)
    indexes_fills_cleaned =np.argwhere(mask_fills_cleaned==True ) +1

    if indexes_fills_cleaned.shape[0] == 0:
        return None
    else:


        tab = np.concatenate \
            ((track[indexes_fills_cleaned - 1], track[indexes_fills_cleaned], track[indexes_fills_cleaned + 1]), axis=1)
        print(tab.shape)
        return tab


if __name__ == '__main__':

    server =True

    if server:
        path = '/home/ftamagnan/lpd_5/lpd_5_cleansed/'
        path_tags = ['/home/herman/lpd/id_lists/tagtraum/tagtraum_Rock.id']

    else:
        path = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
        path_tags = [
            '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
        ]

    macro_iteration(filepath_dataset=path, filepath_tags=path_tags)

