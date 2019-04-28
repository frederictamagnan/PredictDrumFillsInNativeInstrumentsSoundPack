temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'


filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'
filename='output_songs.npy'

from DrumReducerExpander import DrumReducerExpander
encoder=DrumReducerExpander()
import numpy as np
from utils import numpy_drums_save_to_midi
data=np.load(filepath+filename)
print(data.sum())
for i in range(len(data)):
    print(data[i].shape)

    r=data[i,0:2,:,:]
    r=r.reshape((16*2,9))
    print(r.shape)
    r=(r>0.2)*1
    print(r.sum())
    r=encoder.decode(r,no_batch=True)
    r=encoder.decode_808(r,no_batch=True)
    # print(r.sum())
    print(r.shape)
    numpy_drums_save_to_midi(r, temp_filepath, str(i))
    if i>20:
        break

