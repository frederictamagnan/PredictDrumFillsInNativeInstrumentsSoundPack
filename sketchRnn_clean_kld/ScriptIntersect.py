
import numpy as np

def intersect(arr1,arr2,arr3):
    inter1=np.intersect1d(arr1,arr2)

    return np.intersect1d(inter1,arr3)

def return_merge_indices():
    path='/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/indices/'
    prefix='indices_sketchrnn_'
    sufix='_cleaned.pt.npz'
    for i_beta,beta in enumerate([0.01,0.1,1,100,250]):

        beta_list_method=[]
        for i_method,method in enumerate(['Clustering','Diff','Supervised']):
            data=np.load(path+prefix+method+'_'+str(beta)+sufix)
            data=dict(data)
            beta_list_method.append(data['validation'])

        indices_final=intersect(*beta_list_method)
        print(indices_final.shape)


return_merge_indices()






