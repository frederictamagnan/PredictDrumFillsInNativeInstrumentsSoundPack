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

sumi=tr.reshape((tr.shape[0],32*9))
sumi=np.sum(sumi,axis=1)
tr_f=tr[np.where(sumi>10)]
print(sumi.shape)
print(tr_f.shape)



