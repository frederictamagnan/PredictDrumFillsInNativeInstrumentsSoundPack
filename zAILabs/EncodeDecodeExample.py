import pypianoroll as ppr
from pypianoroll import Multitrack,Track
from classifier.utils.DrumReducerExpander import DrumReducerExpander


#Parse a midi file into a Multitrack object
multi=ppr.parse('./temp/test.mid')

drum_track=multi.tracks[0].pianoroll

print(drum_track.shape,"Raw drum track shape")

enc_dec=DrumReducerExpander()

drum_track_reduced=enc_dec.encode(drum_track,no_batch=True)

print(drum_track_reduced.shape," Drum track reduced shape")

drum_track_reduced_16_steps=enc_dec.encode_808(drum_track_reduced,no_batch=True)

print(drum_track_reduced_16_steps.shape ," Drum track reduced shape 16 steps shape")

drum_track_encoded_decoded=enc_dec.decode(enc_dec.decode_808(drum_track_reduced_16_steps,no_batch=True),no_batch=True)

track=Track(pianoroll=drum_track_encoded_decoded,is_drum=True)
multi_encoded_decoded=Multitrack(tracks=[track])


ppr.write(multi_encoded_decoded,"./temp/test_encoded_decoded.mid")
print(drum_track_encoded_decoded.shape, "drum track encoded and then decoded shape")


'''
encode/decode --> 128 instruments /9 instruments

encode_808/decode_808 --> 96 steps / 16 steps


you can also encode or decode "batch of bars" with shape like (100,96,128) without specify the no_batch argument


'''
import numpy as np
batch_drum_track=np.zeros((100,96,128))
batch_drum_track_reduced= enc_dec.encode(batch_drum_track)
print(batch_drum_track_reduced.shape,"batch drum track reduced shape")

