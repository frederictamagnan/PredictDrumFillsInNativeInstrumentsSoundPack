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

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/48/FillsExtractedRnn.npz')
data=dict(data)
for elt in data.keys():
    print(data[elt].shape)
    print(elt)

tr=data['track_array']
#
# # print(tr[0]/128)
# tr=tr>0
# print(np.unique(tr,axis=0).shape,"UNIQUE")
# tr,indices=np.unique(tr,axis=0,return_index=True)
# vae=data['vae'][indices]
# genre=data['genre'][indices]
from DrumReducerExpander import DrumReducerExpander
dec=DrumReducerExpander()
np.random.shuffle(tr)
if 1==1:
    for i in range(len(tr)):
        track=tr[i+300].reshape((4*96,9))

        # rp=tr[i+300,0,:,:]
        # f=tr[i+300,1,:,:]
        # rp2 = tr[i + 300, 2, :, :]

        # track=np.concatenate((rp,f,rp2,rp,f,rp2),axis=0)
        print(track.shape)
        print(track.sum())
        track=np.concatenate((track,track),axis=0)
        track=dec.decode(batch_pianoroll=track,no_batch=True)
        # track=dec.decode_808(batch_pianoroll=track,no_batch=True)
        numpy_drums_save_to_midi(track,'/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/',str(i))

        if i>50:
            break

# minn=7
# for name in ['Supervised07']:
#     data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtracted'+name+'.npz')
#     data=dict(data)
#     for elt in data.keys():
#         print(data[elt].shape)
#         print(elt)
#     tr=data['track_array']
#     tr=tr>0
#     regular=tr[:,0,:,:].reshape((tr.shape[0],16*9))
#     fill=tr[:,1,:,:].reshape((tr.shape[0],16*9))
#     regular_sum=np.sum(regular,axis=1)
#     fill_sum=np.sum(fill,axis=1)
#     indices=np.argwhere(np.logical_and(regular_sum>minn, fill_sum>minn))
#     print(indices.shape,"INDICES")
#     vae=data['vae'][indices[:,0],:,:,:]
#     track_array=data['track_array'][indices[:,0],:,:,:]
#     print(vae.shape,"lol")
#     genre=data['genre'][indices[:,0],:,:]
#     print(genre.shape,"lol")
#     print(track_array.shape,"LOL")
# name='Four'
# # np.savez('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtracted'+name+'_cleaned_v2.npz',vae=vae,track_array=tr,genre=genre)
#
#



