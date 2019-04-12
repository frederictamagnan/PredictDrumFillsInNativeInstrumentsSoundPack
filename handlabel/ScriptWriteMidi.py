
import numpy as np
from DrumReducerExpander import DrumReducerExpander
from pypianoroll import Multitrack,Track
import pypianoroll as ppr

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/validation_velo.npz')
data=dict(data)

tr=data['track_array']
enc=DrumReducerExpander()

tr_d=enc.decode(tr)
tr_dd=enc.decode_808(tr_d)

filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/handlabel/'

def write_midi(array,filepath,filename):
    track = Track(array, is_drum=True)
    multi = Multitrack(tracks=[track])
    ppr.write(multi, filepath + filename)


for i in range(len(tr_dd)):
    write_midi(tr_dd[i],filepath,"track_"+str(i))


