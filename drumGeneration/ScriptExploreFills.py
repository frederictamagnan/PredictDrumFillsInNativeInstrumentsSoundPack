import numpy as np

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/reduced_fills_all_genre.npz')
data=dict(data)
fills=data['fills']
genre=data['genre']

print(genre.shape,fills.shape,genre.sum())