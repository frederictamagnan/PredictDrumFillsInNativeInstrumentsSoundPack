import numpy as np
from utils import numpy_drums_save_to_midi
# data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/reduced_fills_all_genre.npz')
# data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/vae_dataset.npz')
#
# data=dict(data)
# vae=data['vae']
# genre=data['genre']
#
# # print(vae[:10])
# print(genre.shape,vae.shape,genre.sum())

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtractedClustering.npz')
data=dict(data)
for elt in data.keys():
    print(data[elt].shape)
    print(elt)
tr=data['track_array']
tr=tr>0
# from DrumReducerExpander import DrumReducerExpander
# dec=DrumReducerExpander()
# if 1==1:
#     for i in range(len(tr)):
#         track=tr[i+250].reshape((2*16,9))
#         track=dec.decode(batch_pianoroll=track,no_batch=True)
#         track=dec.decode_808(batch_pianoroll=track,no_batch=True)
#         numpy_drums_save_to_midi(track,'/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/',str(i))
#
#         if i>30:
#             break

minn=7
for name in ['Clustering','Supervised','Diff']:
    data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtracted'+name+'.npz')
    data=dict(data)
    tr=data['track_array']
    tr=tr>0
    regular=tr[:,0,:,:].reshape((tr.shape[0],16*9))
    fill=tr[:,1,:,:].reshape((tr.shape[0],16*9))
    regular_sum=np.sum(regular,axis=1)
    fill_sum=np.sum(fill,axis=1)
    indices=np.argwhere(np.logical_and(regular_sum>minn, fill_sum>minn))
    tr_f=tr[indices]
    print(tr.shape)
    print(tr_f.shape)

    for key in data.keys():
        data[key]=data[key][indices]

    np.savez('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtracted'+name+'_cleaned.npz',**data)





