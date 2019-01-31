import numpy as np

# data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/reduced_fills_all_genre.npz')
# data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/vae_dataset.npz')
#
# data=dict(data)
# vae=data['vae']
# genre=data['genre']
#
# # print(vae[:10])
# print(genre.shape,vae.shape,genre.sum())

data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/reduced_fills_all_genre.npz')
data=dict(data)
fills=data['fills']
print(fills.shape)

fills=fills.reshape((19984, 3* 96, 9))
fills=fills.reshape((19984*3*4*4,-1,9))
print(fills.shape)

for i in range(6):
    print(fills[:,i,:].sum())


