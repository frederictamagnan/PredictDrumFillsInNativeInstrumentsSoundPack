import numpy as np

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


    def __init__(self):
        self._drum_type_pitches = DEFAULT_DRUM_TYPE_PITCHES
        self._drum_map = dict(enumerate(DEFAULT_DRUM_TYPE_PITCHES))
        self._inverse_drum_map = dict((pitch, index)
                                      for index, pitches in self._drum_map.items()
                                      for pitch in pitches)





    def encode(self,batch_pianoroll):

        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll size, must be exactly 3"
        batch_encoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],9))
        for i in range(len(self._drum_map)):
             batch_encoded_pianoroll[:,:,i]=np.amax(batch_pianoroll[:,:,self._drum_type_pitches[i]],axis=2)
        return batch_encoded_pianoroll

    def decode(self,batch_pianoroll):

        if len(batch_pianoroll.shape)!=3:
            raise "error in batch pianoroll size, must be exactly 3"

        batch_decoded_pianoroll=np.zeros((batch_pianoroll.shape[0],batch_pianoroll.shape[1],128))
        for i in range(len(self._drum_type_pitches)):
            batch_decoded_pianoroll[:,:,self._drum_map[i][0]]=batch_pianoroll[:,:,i]

        return batch_decoded_pianoroll




if __name__=='__main__':

    lol=np.zeros((256,398,128))
    pr=DrumReducerExpander()

    encoded_lol=pr.encode(lol)
    print(encoded_lol.shape)

    lol=np.zeros((256,1256,9))

    decoded_lol=pr.decode(lol)
    print(decoded_lol.shape)








