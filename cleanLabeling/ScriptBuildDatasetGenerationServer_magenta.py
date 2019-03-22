
# path_tags= [
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Blues.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Country.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Electronic.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Folk.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Jazz.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Latin.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Metal.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_New-Age.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Pop.id', # 8
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Punk.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Rap.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Reggae.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_RnB.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Rock.id', # 13
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_World.id',
#     '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Unknown.id'
# ]



import os
import numpy as np
from pypianoroll import Track, Multitrack
from DrumReducerExpander import DrumReducerExpander

def macro_iteration(filepath_dataset ,max=50000000000000 ,reduced=False ,server=True,whole_dataset=False):

#     fills = np.zeros((1, 3 ,96, 9))
#     vae_array=np.zeros((1,3,32,2))
#     track_array=np.zeros((1,3,16,9))
    vae_array=np.zeros((1,2,32,2))
    track_array=np.zeros((1,2,16,9))
    count = 0

    # ITERATE OVER THE TAG LISTS


    for i, filename in enumerate(os.listdir(filepath_dataset)):

        for npz in os.listdir(filepath_dataset + '/' + filename):
            if 'label.npz' in npz:
                count += 1
    #                         fill = build_generation_dataset(p, npz)
                output=build_generation_dataset(filepath_dataset + '/' + filename,npz)

    #                         if fill is not None:
                if output is not None:
                    vae,track_ar=output
    #                             fill=fill.reshape((fill.shape[0],3*96,128))
    #                             fill = enc.encode(fill)
    #                             fill=fill.reshape((fill.shape[0],3,96,9))
    #                             fills = np.concatenate((fills, fill))
    #                             vae=vae.reshape((vae.shape[0],3,32,2))
                    vae = vae.reshape((vae.shape[0], 2, 32, 2))
                    vae_array=np.concatenate((vae_array,vae))
                    track_array=np.concatenate((track_array,track_ar))
    #                             genre_fill=np.zeros((fill.shape[0],15,1))


#                         if fills.shape[0] >max:
#                             fills = fills[1:]
#                             np.savez("./fills", fills=fills)
#                             return 0

#     fills = fills[1:]
#     genre=genre[1:]
#     np.savez("./reduced_fills_plus_embeddings", fills=fills,genre=genre)

    vae_array=vae_array[1:]
    track_array=track_array[1:]

    np.savez("/home/ftamagna/Documents/_AcademiaSinica/dataset/FillsMagenta",vae=vae_array,track_array=track_array)
    return 0







def build_generation_dataset(p, npz):
    enc=DrumReducerExpander()

    label = dict(np.load(p + '/' + npz))
    label = label['label']
    metadata_dict = dict(np.load(p + '/' + npz.replace('_label','_metadata_training')))
    vae=metadata_dict['vae_embeddings']
    # print(len(label),"LABEL",len(vae),"VAE")
    #     print(label.shape)
    # print(p + '/' + npz.replace('_label.npz', ''))
    multi = Multitrack(p + '/' + npz.replace('_label', ''))
    track = multi.tracks[0].pianoroll
    if track.shape[0] % 96 != 0:
        to_complete_len = 96 - track.shape[0] % 96
        to_complete = np.zeros((to_complete_len, 128))
        track = np.concatenate((track, to_complete))
    track = track.reshape((track.shape[0] // 96, 96, 128))

    # label_previous_next_shape = np.concatenate((label[:-2], label[1:-1], label[2:])).reshape((-1 ,3))
    # label_previous_next_shape = np.concatenate((label[:-1],label[1:])).reshape((-1 ,2))
    #
    #
    # # mask_fills_cleaned =(label_previous_next_shape ==[0, 0, 1]).all(axis=1)
    # mask_fills_cleaned =(label_previous_next_shape ==[0, 1]).all(axis=1)
    #
    # indexes_fills_cleaned =np.argwhere(mask_fills_cleaned==True )

    string = np.array2string(label, precision=0, separator='')[1:-1].replace('.', '').replace(' ', '').replace('\n','')

    indexes_fills_cleaned=allindices(string,'01')

    indexes_fills_cleaned=np.asarray(indexes_fills_cleaned)


    if indexes_fills_cleaned.shape[0] == 0:
        return None
    else:

        # tab=np.concatenate((vae[indexes_fills_cleaned ],vae[indexes_fills_cleaned+1],vae[indexes_fills_cleaned + 2]), axis=1)
        tab=np.concatenate((vae[indexes_fills_cleaned ],vae[indexes_fills_cleaned+1]), axis=1)

        # tab_track = np.concatenate \
        #     ((track[indexes_fills_cleaned ], track[indexes_fills_cleaned+1], track[indexes_fills_cleaned + 2]), axis=1)
        tab_track = np.concatenate \
                ((track[indexes_fills_cleaned ], track[indexes_fills_cleaned+1]), axis=1)
        tab_track=enc.encode(tab_track)
        tab_track=enc.encode_808(tab_track)
        tab_track = tab_track.reshape((-1, 2, 16, 9))
        return tab,tab_track

def allindices(string, sub):
    listindex = []
    offset = 0
    i = string.find(sub, offset)
    while i >= 0:
        listindex.append(i)
        i = string.find(sub, i + 1)
    return listindex



if __name__ == '__main__':

    server =False

    if server:
        path = '/home/ftamagnan/dataset/lpd_5/lpd_5_cleansed/'
        path_tags= [
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Blues.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Country.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Electronic.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Folk.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Jazz.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Latin.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Metal.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_New-Age.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Pop.id', # 8
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Punk.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Rap.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Reggae.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_RnB.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Rock.id', # 13
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_World.id',
    '/home/ftamagnan/dataset/id_lists/tagtraum/tagtraum_Unknown.id'
]


    else:
        path = "/home/ftamagna/Documents/_AcademiaSinica/dataset/magentaDrums/"


    macro_iteration(filepath_dataset=path)

