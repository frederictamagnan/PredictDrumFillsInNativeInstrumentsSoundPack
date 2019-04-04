
import numpy as np

# def intersect(arr1,arr2,arr3):
#     inter1=np.intersect1d(arr1,arr2)
#
#     return np.intersect1d(inter1,arr3)
#
# def return_merge_indices():
#     path='/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/indices/'
#     prefix='indices_sketchrnn_'
#     sufix='_cleaned.pt.npz'
#     for i_beta,beta in enumerate([0.01,0.1,1,100,250]):
#
#         beta_list_method=[]
#         for i_method,method in enumerate(['Clustering','Diff','Supervised']):
#             data=np.load(path+prefix+method+'_'+str(beta)+sufix)
#             data=dict(data)
#             beta_list_method.append(data['validation'])
#
#         indices_final=intersect(*beta_list_method)
#         print(indices_final.shape)
#
#
# return_merge_indices()
#



import numpy as np

filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'
name_raw=['FillsExtractedSupervised.npz','FillsExtractedSupervised03.npz','FillsExtractedRuleBased.npz']

list_arr=[]
for name in name_raw:
    temp=np.load(filepath+name.replace('.npz','_c.npz'))
    temp=dict(temp)
    temp=(temp['track_array']>0)*1
    list_arr.append(temp)

def inter(a,b,c):
    print(a.shape)
    a_=a.reshape((a.shape[0],-1))
    b_ = b.reshape((b.shape[0], -1))
    c_ =c.reshape((c.shape[0], -1))
    print(a.shape,b.shape,c.shape)
    aset = set([tuple(x) for x in a_])
    print(len(aset))

    bset = set([tuple(x) for x in b_])
    cset = set([tuple(x) for x in c_])
    inter=set([x for x in aset & bset & cset])

    abar=aset.difference(inter)
    bbar=bset.difference(inter)
    cbar=cset.difference(inter)

    inter_=np.array([x for x in aset & bset & cset])
    inter_=inter_.reshape((inter_.shape[0],2,16,9))

    for i,elt in enumerate([abar,bbar,cbar]):
        elt=np.array(list(elt))
        elt=elt.reshape((elt.shape[0],2,16,9))
        np.savez(filepath+name_raw[i].replace('.npz','_train.npz'),track_array=elt)

    print(inter_.shape)
    np.savez(filepath+'validation',track_array=inter_)




    # inter = inter.reshape((inter.shape[0], -1))
    # inter_=set([tuple(x) for x in inter])





    #
    # aset_=np.array(aset)
    # bset_ = np.array(bset)
    # cset_ = np.array(cset)
    # abis=np.setdiff1d(aset_, inter)
    # bbis=np.setdiff1d(bset_, inter)
    # cbis=np.setdiff1d(cset_, inter)
    #
    # inter=inter.reshape((inter.shape[0],2,16,9))
    # abis=abis.reshape((abis.shape[0],2,16,9))
    # bbis = bbis.reshape((bbis.shape[0], 2, 16, 9))
    # cbis = abis.reshape((cbis.shape[0], 2, 16, 9))




inter(*list_arr)



