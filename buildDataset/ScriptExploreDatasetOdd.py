newdir="/home/ftamagna/Documents/_AcademiaSinica/dataset/TotalFills/"


import numpy as np

# data1=np.load(newdir+'dataset_NI.npz')
# data2=np.load(newdir+'dataset_odd.npz')

data1=np.load(newdir+'dataset_part_total_NI_1920.npz')
data2=np.load(newdir+'dataset_part_total_odd_1920.npz')


X1 = data1['X']
y_genre1 = data1['y_genre']
y_fills1 = data1['y_fills']
y_bpm1 = data1['y_bpm']
y_dataset1 = data1['y_dataset']

X2 = data2['X']
y_genre2 = data2['y_genre']
y_fills2 = data2['y_fills']
y_bpm2 = data2['y_bpm']
y_dataset2 = data2['y_dataset']



print(y_dataset1[:,0,0].sum()==len(X1))
print(y_dataset2[:,1,0].sum()==len(X2))

print(y_genre1[:,11,0].sum())
print(y_genre2[:,11,0].sum())

total={}

for key in data1.keys():
    print(data1[key].shape,data2[key].shape,key)
    total[key]=np.concatenate((data1[key],data2[key]),axis=0)

y_fills=total['y_fills']
print(y_fills[:,1,0].sum())


# np.savez(newdir+"total",**total)

