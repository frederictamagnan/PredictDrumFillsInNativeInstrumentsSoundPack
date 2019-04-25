import numpy as np
from pypianoroll import Track,Multitrack
import pypianoroll as ppr
from utils import random_file
DEFAULT_DRUM_TYPE_PITCHES_9 = [
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

DEFAULT_DRUM_TYPE_PITCHES_4 = [
    # bass drum
    [36, 35],

    # snare drum
    [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85,45, 29, 41, 61, 64, 84,48, 47, 60, 63, 77, 86, 87,50, 30, 43, 62, 76, 83],

    # closed hi-hat
    [42, 44, 54, 68, 69, 70, 71, 73, 78, 80,51, 52, 53, 59, 82],

    # open hi-hat
    [46, 67, 72, 74, 79, 81,49, 55, 57, 58],

]


DEFAULT_DRUM_TYPE_PITCHES_3 = [
    # bass drum
    [48],

    # snare drum
    [52],
    # closed hi-hat
    [53],

    # open hi-hat
    [55],

]

DEFAULT_DRUM_TYPE_PITCHES_5 = [
    # bass drum
    [48],

    # snare drum
    [52],
    # closed hi-hat
    [54],

    # open hi-hat
    [56],

]

DEFAULT_DRUM_TYPE_PITCHES_6= [
    # bass drum
    [48],

    # snare drum
    [52],

    # closed hi-hat
    [53],

    # open hi-hat
    [55],

    # low tom
    [49],

    # mid tom
    [50],

    # high tom
    [51],

    # crash cymbal
    [56],

    # ride cymbal
    [54]
]









class DrumReducerExpander:


    def __init__(self, offset=False, drumpitches=9):
        if drumpitches==9:
            DEFAULT_DRUM_TYPE_PITCHES=DEFAULT_DRUM_TYPE_PITCHES_9
        elif drumpitches==3:
            DEFAULT_DRUM_TYPE_PITCHES=DEFAULT_DRUM_TYPE_PITCHES_3
        elif drumpitches==5:
            DEFAULT_DRUM_TYPE_PITCHES=DEFAULT_DRUM_TYPE_PITCHES_5
        elif drumpitches==6:
            DEFAULT_DRUM_TYPE_PITCHES=DEFAULT_DRUM_TYPE_PITCHES_6
        else:
            DEFAULT_DRUM_TYPE_PITCHES=DEFAULT_DRUM_TYPE_PITCHES_4

        if offset:
            for i,elt in enumerate(DEFAULT_DRUM_TYPE_PITCHES):
                DEFAULT_DRUM_TYPE_PITCHES_2[i]= [x-24 for x in DEFAULT_DRUM_TYPE_PITCHES[i]]

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
        batch_encoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],len(self._drum_type_pitches)))
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



    def encode_808(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_pianoroll=batch_pianoroll.reshape(batch_pianoroll.shape[0],-1,4,6,batch_pianoroll.shape[2])
        batch_pianoroll_encoded=np.max(batch_pianoroll,axis=3)
        batch_pianoroll_encoded=batch_pianoroll_encoded.reshape(batch_pianoroll_encoded.shape[0],-1,len(self._drum_type_pitches))

        if no_batch:
            batch_pianoroll_encoded=batch_pianoroll_encoded.reshape((batch_pianoroll_encoded.shape[1],batch_pianoroll_encoded.shape[2]))


        return batch_pianoroll_encoded

    def decode_808(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_decoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],6,batch_pianoroll.shape[2]))

        batch_decoded_pianoroll[:,:,0,:]=batch_pianoroll
        batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[0],batch_decoded_pianoroll.shape[1]*batch_decoded_pianoroll.shape[2],batch_decoded_pianoroll.shape[3]))
        if no_batch:
            batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[1],batch_decoded_pianoroll.shape[2]))

        return batch_decoded_pianoroll



    def encode_48(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_pianoroll=batch_pianoroll.reshape(batch_pianoroll.shape[0],-1,12,2,batch_pianoroll.shape[2])
        batch_pianoroll_encoded=np.max(batch_pianoroll,axis=3)
        batch_pianoroll_encoded=batch_pianoroll_encoded.reshape(batch_pianoroll_encoded.shape[0],-1,len(self._drum_type_pitches))

        if no_batch:
            batch_pianoroll_encoded=batch_pianoroll_encoded.reshape((batch_pianoroll_encoded.shape[1],batch_pianoroll_encoded.shape[2]))


        return batch_pianoroll_encoded

    def decode_48(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_decoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],2,batch_pianoroll.shape[2]))

        batch_decoded_pianoroll[:,:,0,:]=batch_pianoroll
        batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[0],batch_decoded_pianoroll.shape[1]*batch_decoded_pianoroll.shape[2],batch_decoded_pianoroll.shape[3]))
        if no_batch:
            batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[1],batch_decoded_pianoroll.shape[2]))

        return batch_decoded_pianoroll






    def encode_4(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_pianoroll=batch_pianoroll.reshape(batch_pianoroll.shape[0],-1,2,12,batch_pianoroll.shape[2])
        batch_pianoroll_encoded=np.max(batch_pianoroll,axis=3)
        batch_pianoroll_encoded=batch_pianoroll_encoded.reshape(batch_pianoroll_encoded.shape[0],-1,len(self._drum_type_pitches))

        if no_batch:
            batch_pianoroll_encoded=batch_pianoroll_encoded.reshape((batch_pianoroll_encoded.shape[1],batch_pianoroll_encoded.shape[2]))


        return batch_pianoroll_encoded

    def decode_4(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_decoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],12,batch_pianoroll.shape[2]))

        batch_decoded_pianoroll[:,:,0,:]=batch_pianoroll
        batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[0],batch_decoded_pianoroll.shape[1]*batch_decoded_pianoroll.shape[2],batch_decoded_pianoroll.shape[3]))
        if no_batch:
            batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[1],batch_decoded_pianoroll.shape[2]))

        return batch_decoded_pianoroll

    def encode_2(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_pianoroll=batch_pianoroll.reshape(batch_pianoroll.shape[0],-1,1,24,batch_pianoroll.shape[2])
        batch_pianoroll_encoded=np.max(batch_pianoroll,axis=3)
        batch_pianoroll_encoded=batch_pianoroll_encoded.reshape(batch_pianoroll_encoded.shape[0],-1,len(self._drum_type_pitches))

        if no_batch:
            batch_pianoroll_encoded=batch_pianoroll_encoded.reshape((batch_pianoroll_encoded.shape[1],batch_pianoroll_encoded.shape[2]))


        return batch_pianoroll_encoded

    def decode_2(self,batch_pianoroll,no_batch=False):

        if no_batch:
            if len(batch_pianoroll.shape)!=2:
                raise "error in batch pianoroll number dimensions, with no batch it must be 2"
            batch_pianoroll=batch_pianoroll.reshape((1,batch_pianoroll.shape[0],batch_pianoroll.shape[1]))


        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll number dimensions, must be exactly 3"

        batch_decoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],24,batch_pianoroll.shape[2]))

        batch_decoded_pianoroll[:,:,0,:]=batch_pianoroll
        batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[0],batch_decoded_pianoroll.shape[1]*batch_decoded_pianoroll.shape[2],batch_decoded_pianoroll.shape[3]))
        if no_batch:
            batch_decoded_pianoroll=batch_decoded_pianoroll.reshape((batch_decoded_pianoroll.shape[1],batch_decoded_pianoroll.shape[2]))

        return batch_decoded_pianoroll


if __name__=='__main__':

    temp_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
    pr=DrumReducerExpander(offset=False)
    print(len(pr._drum_type_pitches))
    filepath,npz=random_file()
    multi=Multitrack(filepath+npz)
    multi_drums=Multitrack(tracks=[Track(multi.tracks[0].pianoroll,is_drum=True)])
    pianoroll=multi.tracks[0].pianoroll

    enc_piano=pr.encode(pianoroll,no_batch=True)
    print(enc_piano.shape)

    enc_piano=pr.encode_808(enc_piano,no_batch=True)
    dec_piano=pr.decode_808(enc_piano,no_batch=True)

    # import sys
    # sys.exit(0)


    dec_piano=pr.decode(dec_piano,no_batch=True)
    track_dec=Track(dec_piano,is_drum=True)
    multi_dec=Multitrack(tracks=[track_dec])

    ppr.write(multi_drums, temp_path + 'track_origin_31.mid')

    ppr.write(multi_dec, temp_path + 'track_enc_dec_31.mid')













