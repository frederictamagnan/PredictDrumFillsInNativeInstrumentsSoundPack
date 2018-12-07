from MultiDrumOneHotEncoding import MultiDrumOneHotEncoding
from pypianoroll import Track, Multitrack
import pypianoroll as ppr
drum_encoding = MultiDrumOneHotEncoding()
from py909.main import drum_track_to_wav


filepath="/home/ftamagna/Documents/_AcademiaSinica/dataset/NI_Drum_Studio_Midi_encoded/MIDI Files/02 Funk/01 Groove 100BPM/01 8th Hat Closed.mid"

multi=ppr.parse(filepath)
multi.binarize()
print(len(multi.tracks[0].pianoroll))
encoded=drum_encoding.multitrack_to_encoded_pianoroll(multi)
print(encoded.sum())

drum_track_to_wav(encoded,"lol")


# pattern = midi.read_midifile(filepath)
#
# for i,elt in enumerate(pattern[0]):
#
#     print(elt)



#THE ENCODING FOR A PIANO ROLL LOOKS OK