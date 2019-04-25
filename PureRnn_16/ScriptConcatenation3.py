filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'

rb='_method_2.npz'
ml='_method_0.npz'
gr='validation.npz'

from scipy import stats
import numpy as np


data=np.load(filepath+rb)
data=dict(data)
rb_array=data['track_array']
print(rb_array.shape)

data=np.load(filepath+ml)
data=dict(data)
ml_array=data['track_array']
print(ml_array.shape)

data=np.load(filepath+gr)
data=dict(data)
gr_array=(data['track_array'][:,1,:,:]>0)*1
print(gr_array.shape)

total=np.stack((gr_array,ml_array,rb_array),axis=1)
print(total.shape)


def testchi2(test):

    rp=test[:,2,:,:].reshape((-1,9))
    f=test[:,1,:,:].reshape((-1,9))

    list_rp = []
    list_f = []

    for i in range(rp.shape[1]):
        list_rp.append(rp[:, i].sum())
        list_f.append(f[:, i].sum())

    del list_f[1]
    del list_rp[1]
    list_rp = np.asarray(list_rp)
    list_f= np.asarray(list_f)
    print(list_f,list_rp)


    o=list_f
    e=list_rp
    arr = np.asarray([o, e])
    chi2, pvalue, dof, expected = stats.chi2_contingency(arr)
    # chi_squared_stat = (np.power(o- e,2) / e).sum()
    # pvalue = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
    #                              df=8)

    return pvalue

print(testchi2(total))

ml_s=ml_array.reshape((-1,9))*1
rb_s=rb_array.reshape((-1,9))*1

gr_s=gr_array.reshape((-1,9))*1


list_ml=[]
list_rb=[]
list_gr=[]

for i in range(9):
    list_ml.append(ml_s[:,i].sum())
    list_rb.append(rb_s[:, i].sum())
    list_gr.append(gr_s[:, i].sum())



print(list_ml,list_rb,list_gr)
# import matplotlib.pyplot as plt
#
# fig, axs = plt.subplots(3, 1,sharey=True)
# axs[0].bar(name_pitches, list_p)
# axs[1].bar(name_pitches, list_a)
# axs[2].bar(name_pitches, list_d)
# plt.show()

# import pandas as pd
# #tom
# data=pd.DataFrame()
# namei=['gt',"ml", "rb"]
# for i in range(3):
#     print(namei[i] )
#     toms=total[:,i,:,4:7]
#     toms=toms.reshape((toms.shape[0],16*3))
#     toms=np.sum(toms,axis=1)
#     bass=np.sum(total[:,i,:,0],axis=1)
#     cymbal=np.concatenate((total[:,i,:,2:4],total[:,i,:,7:9]),axis=2)
#     cymbal=cymbal.reshape((cymbal.shape[0],16*4))
#     cymbal=np.sum(cymbal,axis=1)
#     data['cymbals of '+namei[i]]=cymbal
#     data['bass of ' + namei[i]] = bass
#     data['toms of ' + namei[i]] = toms
#
#     print(cymbal.shape,bass.shape,toms.shape)
#
#     # name=['cymbal',"bass","toms"]
#     # for j,elt in enumerate([cymbal,bass,toms]):
#     #     unique_elements, counts_elements = np.unique(elt, return_counts=True)
#     #     print("Frequency of unique values of the said array : "+name[j])
#     #     print(np.asarray((unique_elements, counts_elements)))
#
#
# print(data.keys())
# import matplotlib.pyplot as plt
# import seaborn as sns
# color_name=['red','skyblue','green']
# f, axes = plt.subplots(3, 3, figsize=(7, 7), sharex=True,sharey=True)
# for i,elt in enumerate(['cymbals',"bass","toms"]):
#     for j,elt2 in enumerate(['gt',"ml", "rb"]):
#         sns.distplot(data[elt+' of '+elt2],color=color_name[j],ax=axes[j,i],label= elt+' /'+ elt2)
#     # sns.plt.legend()
#     print("lol")
# plt.show()
# plt.clf()


# data=pd.DataFrame()
namei=['gt',"ml", "rb"]
list_metrics=[]
for i in range(3):
    row=[]
    for j in range(9):
        ins=total[:,i,:,j].reshape((total.shape[0],16))
        amount=np.sum(ins,axis=1)
        row.append([np.mean(amount),np.std(amount),np.min(amount),np.max(amount)])
    # print(row)
    row=np.asarray(row)
    list_metrics.append(row)
for i in list_metrics:
    print("i---------")
    print(i[:,1])