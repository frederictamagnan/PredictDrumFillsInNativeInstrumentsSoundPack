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


class MultiDrumOneHotEncoding():
    def __init__(self):
        self._drum_type_pitches = DEFAULT_DRUM_TYPE_PITCHES
        self._drum_map = dict(enumerate(DEFAULT_DRUM_TYPE_PITCHES))
        self._inverse_drum_map = dict((pitch, index)
                                      for index, pitches in self._drum_map.items()
                                      for pitch in pitches)

    def encode_drum(self, pitches_in):
        nonzero = np.where(pitches_in == 1)[0]
        ret = np.zeros(len(self._drum_type_pitches))
        for reduced, pitches in self._drum_map.items():
            for p in pitches:
                if p in nonzero:
                    ret[reduced] = 1
                    break
        return ret

    def encode_drum_velocity(self, pitches_in):
        nonzero = np.where(pitches_in >0)[0]
        ret = np.zeros(len(self._drum_type_pitches))
        for reduced, pitches in self._drum_map.items():
            for p in pitches:
                if p in nonzero:
                    ret[reduced] = pitches_in[p]
                    break
        return ret



    def multitrack_to_encoded_pianoroll(self, multitrack):

        pianoroll=multitrack.tracks[0].pianoroll
        ret=np.zeros((pianoroll.shape[0],len(self._drum_type_pitches)))

        for i, elt in enumerate(pianoroll):

            encoded_elt=self.encode_drum(elt)

            ret[i,:]=encoded_elt

        return ret

    def multitrack_to_encoded_pianoroll_velocity(self, multitrack):

        pianoroll=multitrack.tracks[0].pianoroll
        ret=np.zeros((pianoroll.shape[0],len(self._drum_type_pitches)))

        for i, elt in enumerate(pianoroll):

            encoded_elt=self.encode_drum_velocity(elt)

            ret[i,:]=encoded_elt

        return ret










