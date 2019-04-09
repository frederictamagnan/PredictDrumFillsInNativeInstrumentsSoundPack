from utils import numpy_drums_save_to_midi
import numpy as np
from DrumReducerExpander import DrumReducerExpander

dataset_array='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'
name_array='validation.npz'
temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
array = np.load(dataset_array + name_array)

array=dict(array)
array=1*(array['track_array'][:30]>0)

dumb=np.zeros(array.shape)
dumb[:,0,9,4]=1
dumb[:,0,11,5]=1
dumb[:,0,13,6]=1
dumb[:,0,15,7]=1

dumb=dumb+array
new = np.concatenate((array[:,0,:,:],array[:,0,:,:],array[:,0,:,:],dumb[:,0,:,:],array[:,0,:,:],array[:,0,:,:],array[:,0,:,:],dumb[:,0,:,:]),axis=1)
print(new.shape)
enc=DrumReducerExpander()
new=enc.decode(new)
new=enc.decode_808(new)


for i in range(len(new)):
    numpy_drums_save_to_midi(new[i], temp_filepath, "sample_" + str(i) + "_dumb")