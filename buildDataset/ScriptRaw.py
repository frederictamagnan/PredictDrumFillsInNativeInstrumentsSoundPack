import numpy as np
path='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/RawFiles/NI_raw.npz'
array=dict(np.load(path))['X']
print(array.shape)