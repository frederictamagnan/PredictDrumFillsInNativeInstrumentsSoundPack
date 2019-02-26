from VaeEncoderDecoder_808_9 import VaeEncoderDecoder
import pypianoroll as ppr
from pypianoroll import Track,Multitrack
from utils import numpy_drums_save_to_midi
from DrumsDataset import EmbeddingsDataset
from DrumReducerExpander import DrumReducerExpander



import numpy as np
decoderVAE = VaeEncoderDecoder()


data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/bigsupervised.npz')

tr=data['track_array'][0]

print(tr.shape)
tr_emb=decoderVAE.encode_to_embeddings(tr)


mu=np.zeros((1000,32,1))
# print(mu)
sigma=np.ones((1000,32,1))

lol=np.concatenate((mu,sigma),axis=2)
print(lol.shape,"lol")

decoder= DrumReducerExpander(drumpitches=9)

# dataset=EmbeddingsDataset(lol)
#
# print(dataset)



drums_reduced=decoderVAE.decode_to_reduced_drums(lol)


# print(drums_reduced)
print(drums_reduced.shape)
drums_reduced=drums_reduced>0.75
filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'

# np.save(filepath+"generated_with_normal",drums_reduced)

expanded_drums=decoder.decode(drums_reduced)
expanded_drums=decoder.decode_808(expanded_drums)
#
for i in range(len(expanded_drums)):
    numpy_drums_save_to_midi(np.concatenate((expanded_drums[i],expanded_drums[i])),filepath,str(i))



