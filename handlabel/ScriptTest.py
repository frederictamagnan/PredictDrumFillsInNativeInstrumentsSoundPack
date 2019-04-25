import numpy as np
filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumDetection/'
name='validation_velo.npz'
data=np.load(filepath+name)
data=dict(data)
data=data['track_array']
print(data.shape)