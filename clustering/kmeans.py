import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import random_file
from sklearn.cluster import KMeans
import random
from pypianoroll import Track,Multitrack
random.seed(12)
import pypianoroll as ppr
path,npz=random_file()
mt=dict(np.load(path+'/'+npz.replace('.npz','_metadata_training.npz')))

X=mt['velocity_metadata']
# X=mt['vae_embeddings']

print(X.shape)
tsne = TSNE(n_components=2, random_state=0)
# X_trans= tsne.fit_transform(X)
random_state = 170
km=KMeans(n_clusters=6, random_state=random_state)
y_pred = km.fit_predict(X)
# plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y_pred)
# plt.show()

path_temp = '/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'

for i in range(6):
    tab_index_bar=np.argwhere(y_pred==i)
    index_bar=tab_index_bar[0][0]

    print(index_bar,type(index_bar))
    multi=Multitrack(path+npz)
    print(multi.tracks[0].pianoroll.shape)
    pianoroll=multi.tracks[0].pianoroll[index_bar*96:(index_bar+1)*96,:]
    multi=Track(pianoroll,is_drum=True)
    multi=Multitrack(tracks=[multi])
    ppr.write(multi, filepath=path_temp + 'multi_cluster_'+str(i)+'_len_'+str(len(tab_index_bar)))
