import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
import sys
data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtractedSupervised.npz')

def unison_shuffled_copies(a, b,c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p],c[p]

print(data['vae'].shape)

tr=data['track_array']
max_axis=np.max(tr,axis=2)
std_axis=np.std(tr,axis=2,dtype=np.float64)
mean_axis=np.mean(tr,axis=2,dtype=np.float64)

velocity_metadata=np.concatenate([max_axis,std_axis,mean_axis],axis=2)
velocity_metadata=velocity_metadata.reshape((velocity_metadata.shape[0],velocity_metadata.shape[1],-1))
print(velocity_metadata.shape,"velshape")
X_0_mu=velocity_metadata[:,0,4:]
X_1_mu=velocity_metadata[:,1,4:]

genre=np.concatenate((data['genre'],data['genre']))
print(genre.shape,'gere')
y=np.concatenate((np.zeros(X_0_mu.shape[0]),np.ones(X_1_mu.shape[0])))
print(y.shape,y.sum())
X=np.concatenate((X_0_mu,X_1_mu))
print(X.shape)

target_ids = range(2)
target_names='regular drum','drum fill'

X,y,genre=unison_shuffled_copies(X,y,genre)


# plt.figure(figsize=(5, 6))
colors = 'r', 'g'

style=[
'Blues.id',
'Country.id',



'Electronic.id',
'Folk.id',
'Jazz.id',
'Latin.id',
'Metal.id',
'New-Age.id',
'Pop.id',
'Punk.id',
'Rap.id',
'Reggae.id',
'RnB.id',
    'Rock.id',
'World.id',
'Unknown.id'

    ]

for j,elt in enumerate(style):

    X_=X[genre[:,j,0]==1,:]
    y_ = y[genre[:, j,0] == 1]
    X_mu=X_[0:500]
    y_=y_[0:500]

    tsne = TSNE(n_components=2, random_state=0)
    X_mu_2d = tsne.fit_transform(X_mu)
    # X_mu_2d=X_mu
    print(X_mu_2d.shape)
    # X_std_2d = SelectKBest(chi2, k=2).fit_transform(X_mu+0.5, y)


    for i, c, label in zip(target_ids, colors,target_names):
        plt.scatter(X_mu_2d[y_== i, 0], X_mu_2d[y_ == i, 1], c=c, label=label)
        plt.title(elt)
    plt.legend()
    plt.show()



