from utils import random_file
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import path_to_midi
random.seed(5)

path,npz=random_file()
path_to_midi(path,npz)
print(path,npz[:-4])

label=np.load(path+npz[:-4]+"_label.npz")
metrics=np.load(path+npz[:-4]+"_metrics_training.npz")

metrics=dict(metrics)
label=dict(label)

print(metrics['velocity_metrics'].shape)
print(label['label'].shape)

vel=metrics['velocity_metrics']
lab=label['label']

# views on each column
# j=1
#
# list_index=[]
# for i in range(36):
#     if vel[:, i].mean() != 0:
#         list_index.append(i)
#
# for i,index in enumerate(list_index):
#         plt.subplot(len(list_index)+1, 1, i+1)
#         plt.plot(vel[:, index]>vel[:, index].max()/2)
#
# plt.subplot(len(list_index)+1,1,len(list_index)+1)
# plt.ylabel('label')
# plt.plot(lab)
# plt.show()


print(vel.shape)
plt.subplot(2, 1, 1)
plt.plot(np.power(vel,2).sum(axis=1))

plt.subplot(2,1,2)
plt.ylabel('label')
plt.plot(lab)
plt.show()
