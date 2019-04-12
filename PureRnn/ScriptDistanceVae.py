filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'

filename=['_method_0.npz_vae.npy','_method_2.npz_vae.npy',"validation.npz_vae.npy"]
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean

for elt in filename:
    sum=0
    data=np.load(filepath+elt)
    print(data.shape)
    data=data[:,:32]
    for i in range(len(data)):
        for j in range(i,len(data)):
            # sum=norm(data[i]-data[j])+sum
            sum=euclidean(data[i],data[j])+sum


    print(sum)
# i=0
# for j in range(10):
#     i=i+j
# print(j)
