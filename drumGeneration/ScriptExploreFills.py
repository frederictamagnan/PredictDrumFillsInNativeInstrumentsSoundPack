import numpy as np

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/fills_reduced.npz')
data=dict(data)
data=data['fills']

print(data.shape)