import numpy as np

# r=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_regression_t.npy')
data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/1502supervised.npz')
data=data['track_array']
print(data.shape,"data shape")
p=data[:,0,:,:]
# p=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_regression_previous_t.npy')
a=data[:,1,:,:]

# n=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/evaluation/'+'generated_with_normal.npy')

name_pitches = ['bass drum','snare drum','closed hi-hat','open hi-hat','low tom','mid tom','high tom','crash cymbal','ride cymbal']

print(p.shape,a.shape)

p=p.reshape((-1,9))*1
a=a.reshape((-1,9))*1

list_p=[]
list_a=[]

for i in range(9):
    list_p.append(p[:,i].sum())
    list_a.append(a[:, i].sum())



print(list_p,list_a)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)
axs[0].bar(name_pitches, list_p)
axs[1].bar(name_pitches, list_a)
plt.show()




