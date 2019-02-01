from VaeEncoderDecoder_808_9 import VaeEncoderDecoder
import pypianoroll as ppr
from pypianoroll import Track,Multitrack
from utils import numpy_drums_save_to_midi
from DrumsDataset import EmbeddingsDataset
from DrumReducerExpander import DrumReducerExpander



import numpy as np


mu=np.zeros((200,32,1))
# print(mu)
sigma=np.ones((200,32,1))

lol=np.concatenate((mu,sigma),axis=2)


decoder= DrumReducerExpander(drumpitches=9)

# dataset=EmbeddingsDataset(lol)
#
# print(dataset)

decoderVAE = VaeEncoderDecoder()

drums_reduced=decoderVAE.decode_to_reduced_drums(lol)


# print(drums_reduced)
print(drums_reduced.shape)
drums_reduced=drums_reduced>0.75

expanded_drums=decoder.decode(drums_reduced)
expanded_drums=decoder.decode_808(expanded_drums)
filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'

for i in range(len(expanded_drums)):
    numpy_drums_save_to_midi(np.concatenate((expanded_drums[i],expanded_drums[i])),filepath,str(i))



