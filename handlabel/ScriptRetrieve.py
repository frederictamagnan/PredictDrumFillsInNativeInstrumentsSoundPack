
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

def macro_iteration(filepath_dataset, filepath_tags ,max=50000000000000 ,reduced=False ,server=True,whole_dataset=False):

#     fills = np.zeros((1, 3 ,96, 9))
#     vae_array=np.zeros((1,3,32,2))
#     track_array=np.zeros((1,3,16,9))
    vae_array=np.zeros((1,4,32,2))
    track_array=np.zeros((1,4,96,9))
    # real_track_array=np.zeros((1,3,196,128))
    count = 0
    genre=np.zeros((1,16,1))
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
                    if 'label_rnn.npz' in npz:
                        count += 1
#                         fill = build_generation_dataset(p, npz)
                        output=build_generation_dataset(p,npz)

#                         if fill is not None:
                        if output is not None:
                            vae,track_ar=output
#                             fill=fill.reshape((fill.shape[0],3*96,128))
#                             fill = enc.encode(fill)
#                             fill=fill.reshape((fill.shape[0],3,96,9))
#                             fills = np.concatenate((fills, fill))
#                             vae=vae.reshape((vae.shape[0],3,32,2))
                            vae = vae.reshape((vae.shape[0], 4, 32, 2))
                            vae_array=np.concatenate((vae_array,vae))
                            track_array=np.concatenate((track_array,track_ar))
                            # real_track_array=np.concatenate((real_track_array,tab_ar_o))
#                             genre_fill=np.zeros((fill.shape[0],15,1))
                            genre_fill=np.zeros((vae.shape[0],16,1))
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
    track_array=track_array[1:]
    genre=genre[1:]
    # real_track_array=real_track_array[1:]
    np.savez("./FillsExtractedHand",vae=vae_array,genre=genre,track_array=track_array)
    return 0







def build_generation_dataset(p, npz):
    enc=DrumReducerExpander()
    ref=np.load('/home/ftamagnan/dataset/handlabel.npy')

    metadata_dict = dict(np.load(p + '/' + npz.replace('_label_rnn','_metadata_training')))
    vae=metadata_dict['vae_embeddings']
    multi = Multitrack(p + '/' + npz.replace('_label_rnn', ''))
    track = multi.tracks[0].pianoroll
    if track.shape[0] % 96 != 0:
        to_complete_len = 96 - track.shape[0] % 96
        to_complete = np.zeros((to_complete_len, 128))
        track = np.concatenate((track, to_complete))
    track = track.reshape((track.shape[0] // 96, 96, 128))

    track_enc=enc.encode(track)
    track_enc=enc.encode_808(track_enc)

    compare_track=np.concatenate((track_enc[:-2],track_enc[1:-1],track_enc[2:]))
    indexes_fills_cleaned=[]

    for i in range(len(track_enc)):
        for j in range(len(ref)):
            if np.equal(compare_track[i],ref[j])
                indexes_fills_cleaned.append(i)
    indexes_fills_cleaned=indexes_fills_cleaned+1
    indexes_fills_cleaned=np.array(indexes_fills_cleaned)


    if indexes_fills_cleaned.shape[0] == 0:
        return None
    else:

        tab=np.concatenate((

            vae[indexes_fills_cleaned -3],
            vae[indexes_fills_cleaned-2],
            vae[indexes_fills_cleaned-1],
            vae[indexes_fills_cleaned ]), axis=1)
        # tab=np.concatenate((vae[indexes_fills_cleaned ],vae[indexes_fills_cleaned+1]), axis=1)

        tab_track_o = np.concatenate \
            ((track[indexes_fills_cleaned -3 ], track[indexes_fills_cleaned-2], track[indexes_fills_cleaned -1], track[indexes_fills_cleaned ]), axis=1)
        # tab_track = np.concatenate \
        #         ((track[indexes_fills_cleaned ], track[indexes_fills_cleaned+1]), axis=1)
        tab_track=enc.encode(tab_track_o)
        # tab_track=enc.encode_808(tab_track)
        tab_track = tab_track.reshape((-1, 4, 96, 9))
        # tab_track_o=tab_track_o.reshape((-1,4,96,128))
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

    server =True

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
        path = '//home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_debug/'
        path_tags = [
            '/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id',
        ]

    macro_iteration(filepath_dataset=path, filepath_tags=path_tags,whole_dataset=True)

