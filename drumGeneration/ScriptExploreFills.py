import numpy as np

# data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/reduced_fills_all_genre.npz')
data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/reduced_fills_plus_embeddings.npz')

data=dict(data)
vae=data['vae']
genre=data['genre']

print(genre.shape,vae.shape,genre.sum())