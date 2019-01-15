import numpy as np

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/fills.npz')
data=dict(data)
data=data['fills']

print(data.shape)