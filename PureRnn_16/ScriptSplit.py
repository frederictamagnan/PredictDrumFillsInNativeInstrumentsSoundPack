import numpy as np

filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/48/'
name='FillsExtractedHand_c.npz'

data=np.load(filepath+name)

data=dict(data)
data=data['track_array']

testing_ratio=0.95

n_training=int(data.shape[0]*testing_ratio)
n_testing=len(data)-n_training

np.save(filepath+'train',data[:n_training])
np.save(filepath+'test',data[n_training:])