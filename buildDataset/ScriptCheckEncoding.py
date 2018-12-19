import pretty_midi
import numpy as np
newdir="/home/ftamagna/Documents/_AcademiaSinica/dataset/TotalFills/"

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

_drum_map = dict(enumerate(DEFAULT_DRUM_TYPE_PITCHES))
_inverse_drum_map =  dict((pitch, index)
                                  for index, pitches in _drum_map.items()
                                  for pitch in pitches)


class MultiDrumOneHotEncoding():
    def __init__(self):
        self._drum_type_pitches = DEFAULT_DRUM_TYPE_PITCHES
        self._drum_map = dict(enumerate(DEFAULT_DRUM_TYPE_PITCHES))
        self._inverse_drum_map = dict((pitch, index)
                                      for index, pitches in _drum_map.items()
                                      for pitch in pitches)

    def encode_drum(self, pitches_in):
        nonzero = np.where(pitches_in == 1)[0] + 24
        ret = np.zeros(len(self._drum_type_pitches))
        for reduced, pitches in _drum_map.items():
            for p in pitches:
                if p in nonzero:
                    ret[reduced] = 1
                    break
        return ret

    def decode_drum(self, pitches_out):
        ret = np.zeros(84)
        for reduced, p in enumerate(pitches_out):
            if p == 1:
                ret[self._drum_type_pitches[reduced][0] - 24] = 1
        return ret


drum_encoding = MultiDrumOneHotEncoding()
def checkDrumEmpty(track):
    compare = (track == np.zeros(track.shape))
    count = np.size(compare) - np.count_nonzero(compare)
    if count > 3:
        return True
    return False


data = np.load(newdir+'dataset_odd_encoded_-24.npz')
train_x_phr_np=data['X']
print(train_x_phr_np.shape)
check = [checkDrumEmpty(train_x_phr_np[i]) for i in range(train_x_phr_np.shape[0])]
train_x_drum_clean = train_x_phr_np[check]
print(train_x_drum_clean.shape)
print(train_x_drum_clean.shape)
nz = np.nonzero(train_x_drum_clean)
if nz[2].size > 0:
    print('min: {}, max: {}'.format(nz[2].min(), nz[2].max()))

for i in range(84):
    places = nz[0][np.where(nz[2] == i)]
    print('[{}]{}: {}'.format(
        i,
        pretty_midi.note_number_to_drum_name(i + 24),
        places.shape[0]))



train_x_drum_clean_reduced = np.zeros((
    train_x_drum_clean.shape[0],
    train_x_drum_clean.shape[1],
    len(DEFAULT_DRUM_TYPE_PITCHES),
))

for bar_i, bar in enumerate(train_x_drum_clean):
    print('converting...{}/{}'.format(bar_i + 1, train_x_drum_clean.shape[0]), end="\r")
    for beat_i, beat in enumerate(bar):
        train_x_drum_clean_reduced[bar_i][beat_i] = drum_encoding.encode_drum(beat)

print()
print(train_x_drum_clean_reduced.shape)
