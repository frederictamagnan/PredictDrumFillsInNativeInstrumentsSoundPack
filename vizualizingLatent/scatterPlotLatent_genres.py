import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data=np.load('./../data/latentDataset_genres.npz')



X_mu=data['X'][:500,:,0]
X_std=data['X'][:500,:,1]

y=data['y'][:500]

print(X_mu.shape,X_std.shape,y.shape)


target_names= ["Pop","Funk","Jazz","Hard","Metal","Blues & Country","Blues Rock","Ballad","Indie Rock","Indie Disco","Punk Rock"]

print(type(target_names))

target_ids = range(len(target_names))



plt.figure(figsize=(20, 20))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple','#eeefff'





tsne = TSNE(n_components=2, random_state=0)
X_mu_2d = tsne.fit_transform(X_mu)
X_std_2d=tsne.fit_transform(X_std)


for i, c, label in zip(target_ids, colors,target_names ):
    plt.scatter(X_std[y == i, 0], X_std[y == i, 1], c=c, label=label)
plt.legend()
plt.show()
print(X_mu_2d[y==0,0].shape)

# for j in range(0,32):
#     for w in range(j+1,32):
#
#         for i, c, label in zip(target_ids, colors,target_names ):
#             plt.scatter(X_mu[y == i,j], X_mu[y == i,w], c=c, label=label)
#         plt.legend()
#         plt.show()

