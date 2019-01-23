
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

def macro_iteration(filepath_dataset, filepath_tags ,max=50000000000000 ,reduced=False ,server=True):

    enc=DrumReducerExpander()
#     fills = np.zeros((1, 3 ,96, 9))
    vae_array=np.zeros((1,3,32))
    count = 0
    genre=np.zeros((1,15,1))
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
#                         fill = build_generation_dataset(p, npz)
                        vae=build_generation_dataset(p,npz)

#                         if fill is not None:
                        if vae is not None:
#                             fill=fill.reshape((fill.shape[0],3*96,128))
#                             fill = enc.encode(fill)
#                             fill=fill.reshape((fill.shape[0],3,96,9))
#                             fills = np.concatenate((fills, fill))
                            vae=vae.reshape((vae.shape[0],3,32))
                            vae_array=np.concatenate((vae_array,vae))
#                             genre_fill=np.zeros((fill.shape[0],15,1))
                            genre_fill=np.zeros((vae.shape[0],15,1))
                            genre_fill[:,tag_i,0]=1
                            genre=np.concatenate((genre,genre_fill))

#                         if fills.shape[0] >max:
#                             fills = fills[1:]
#                             np.savez("./fills", fills=fills)
#                             return 0
    
#     fills = fills[1:]
#     genre=genre[1:]
#     np.savez("./reduced_fills_plus_embeddings", fills=fills,genre=genre)

    vae_array=vae_array[1:]
    genre=genre[1:]
    np.savez("./reduced_fills_plus_embbedings",vae=vae_array,genre=genre)
    return 0







def build_generation_dataset(p, npz):
    label = dict(np.load(p + '/' + npz))
    label = label['label']
    metrics_dict = dict(np.load(p + '/' + npz.replace('_label','_metrics_training')))
    vae=metrics_dict['vae_embeddings']
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

        tab=np.concatenate((vae[indexes_fills_cleaned - 1],vae[indexes_fills_cleaned],vae[indexes_fills_cleaned + 1]), axis=1)
        print(tab.shape,"tab vae shape")
#         tab = np.concatenate \
#             ((track[indexes_fills_cleaned - 1], track[indexes_fills_cleaned], track[indexes_fills_cleaned + 1]), axis=1)
#         print(tab.shape)
        return tab


if __name__ == '__main__':

    server =True

    if server:
        path = '/home/ftamagnan/dataset/lpd_5/lpd_5_cleansed/'
        path_tags= [
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


    else:
        path = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
        path_tags = [
            '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
        ]

    macro_iteration(filepath_dataset=path, filepath_tags=path_tags)

