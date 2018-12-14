import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
import sys
data=np.load('./../data/latentDataset_genre_fill.npz')



X_mu_raw=data['X'][:,:,0]
X_std_raw=data['X'][:,:,1]



y_raw=data['y']



target_ids = range(2)
target_names='regular drum','drum fill'




# plt.figure(figsize=(5, 6))
colors = 'r', 'g'

style= ["Pop","Funk","Jazz","Hard","Metal","Blues & Country","Blues Rock","Ballad","Indie Rock","Indie Disco","Punk Rock"]


for j,elt in enumerate(style):


    # X_mu=X_mu_raw[indexes]
    print(elt)
    y=y_raw[y_raw[:,0,0]==j,1,0]
    print(y[y==0].shape[0],y[y==1].shape[0])
    X_std=X_std_raw[y_raw[:,0,0]==j]

    X_std=X_std[0:500]
    y=y[0:500]

    tsne = TSNE(n_components=2, random_state=0)
    X_std_2d = tsne.fit_transform(X_std)
    print(X_std_2d.shape)
    # X_std_2d = SelectKBest(chi2, k=2).fit_transform(X_mu+0.5, y)







    for i, c, label in zip(target_ids, colors,target_names):
        plt.scatter(X_std_2d[y== i, 0], X_std_2d[y == i, 1], c=c, label=label)
        plt.title(elt)
    plt.legend()
    plt.show()



