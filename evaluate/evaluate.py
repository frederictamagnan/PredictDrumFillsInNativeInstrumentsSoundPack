import numpy as np

r=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_regression_t.npy')
p=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_regression_previous_t.npy')
n=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_normal.npy')

name_pitches = ['bass drum','snare drum','closed hi-hat','open hi-hat','low tom','mid tom','high tom','crash cymbal','ride cymbal']

print(r.shape,p.shape,n.shape)

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

fig, axs = plt.subplots(1, 1)
axs.bar(name_pitches, list_p)
plt.show()
fig, axs = plt.subplots(1, 1)
axs.bar(name_pitches, list_n)
plt.show()
fig, axs = plt.subplots(1, 1)
axs.bar(name_pitches, list_r)
plt.show()




