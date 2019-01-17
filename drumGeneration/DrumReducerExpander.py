import numpy as np
from pypianoroll import Track,Multitrack
import pypianoroll as ppr
from utils import random_file
DEFAULT_DRUM_TYPE_PITCHES = [
    # bass drum
    [36, 35],

    # snare drum
    [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],

    # closed hi-hat
    [42, 44, 54, 68, 69, 70, 71, 73, 78, 80],

    # open hi-hat
    [46, 67, 72, 74, 79, 81],

    # low tom
    [45, 29, 41, 61, 64, 84],

    # mid tom
    [48, 47, 60, 63, 77, 86, 87],

    # high tom
    [50, 30, 43, 62, 76, 83],

    # crash cymbal
    [49, 55, 57, 58],

    # ride cymbal
    [51, 52, 53, 59, 82]
]



class DrumReducerExpander:


    def __init__(self, offset=False):


        if offset:
            for i,elt in enumerate(DEFAULT_DRUM_TYPE_PITCHES):
                DEFAULT_DRUM_TYPE_PITCHES[i]= [x-24 for x in DEFAULT_DRUM_TYPE_PITCHES[i]]

        self._drum_type_pitches = DEFAULT_DRUM_TYPE_PITCHES
        self._drum_map = dict(enumerate(DEFAULT_DRUM_TYPE_PITCHES))
        self._inverse_drum_map = dict((pitch, index)
                                      for index, pitches in self._drum_map.items()
                                      for pitch in pitches)







    def encode(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"
        batch_encoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],9))
        for i in range(len(self._drum_map)):
             batch_encoded_pianoroll[:,:,i]=np.amax(batch_pianoroll[:,:,self._drum_type_pitches[i]],axis=2)


        if no_batch:
            batch_encoded_pianoroll=batch_encoded_pianoroll.reshape((batch_encoded_pianoroll.shape[1],batch_encoded_pianoroll.shape[2]))

        return batch_encoded_pianoroll

    def decode(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))



        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_decoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],128))
        for i in range(len(self._drum_type_pitches)):
            batch_decoded_pianoroll[:,:,self._drum_map[i][0]]=batch_pianoroll[:,:,i]


        if no_batch:
            batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[1],batch_decoded_pianoroll.shape[2]))

        return batch_decoded_pianoroll




if __name__=='__main__':

    temp_path='./temp/'
    pr=DrumReducerExpander(offset=False)
    filepath,npz=random_file()
    multi=Multitrack(filepath+npz)
    multi_drums=Multitrack(tracks=[Track(multi.tracks[0].pianoroll,is_drum=True)])
    pianoroll=multi.tracks[0].pianoroll

    enc_piano=pr.encode(pianoroll,no_batch=True)
    dec_piano=pr.decode(enc_piano,no_batch=True)
    track_dec=Track(dec_piano,is_drum=True)
    multi_dec=Multitrack(tracks=[track_dec])

    ppr.write(multi_drums, temp_path + 'track_origin.mid')

    ppr.write(multi_dec, temp_path + 'track_enc_dec.mid')













