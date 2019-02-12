import numpy as np

r=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_regression_s2.npy')
p=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_regression_previous.npy')
n=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_normal.npy')

name_pitches = ['bass drum','snare drum','closed hi-hat','open hi-hat','low tom','mid tom','high tom','crash cymbal','ride cymbal']

print(r.shape,p.shape)

r=r.reshape((-1,9))*1
p=p.reshape((-1,9))*1
n=n.reshape((-1,9))*1
list_r=[]
list_p=[]
list_n=[]
for i in range(9):
    list_r.append(r[:,i].sum())
    list_p.append(p[:, i].sum())
    list_n.append(n[:, i].sum())


print(list_r,list_p,list_n)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1)
axs[0].bar(name_pitches, list_p)
axs[1].bar(name_pitches, list_n)
axs[2].bar(name_pitches, list_r)


plt.show()

