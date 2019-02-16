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

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/1502supervised.npz')
data=dict(data)
for elt in data.keys():
    print(data[elt].shape)
    print(elt)
tr=data['track_array']
from DrumReducerExpander import DrumReducerExpander
dec=DrumReducerExpander()
for i in range(len(tr)):
    track=tr[i+0].reshape((2*16,9))
    track=dec.decode(batch_pianoroll=track,no_batch=True)
    track=dec.decode_808(batch_pianoroll=track,no_batch=True)
    numpy_drums_save_to_midi(track,'/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/',str(i))

    if i>10:
        break




