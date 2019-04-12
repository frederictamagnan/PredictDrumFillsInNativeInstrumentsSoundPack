
import numpy as np
from DrumReducerExpander import DrumReducerExpander
from pypianoroll import Multitrack,Track
import pypianoroll as ppr

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumDetection/validation_velo.npz')
data=dict(data)

tr=data['track_array']
print(tr.shape)
enc=DrumReducerExpander()



filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/handlabel/'

def write_midi(array,filepath,filename):

    concat=np.concatenate((array[0,:,:],array[0,:,:],array[1,:,:],array[2,:,:],array[2,:,:]),axis=0)
    concat = enc.decode(concat,no_batch=True)
    concat = enc.decode_808(concat,no_batch=True)
    track = Track(concat, is_drum=True)
    multi = Multitrack(tracks=[track])
    ppr.write(multi, filepath + filename)


for i in range(len(tr)):
    write_midi(tr[i],filepath,"track_"+str(i))
    print(i)


