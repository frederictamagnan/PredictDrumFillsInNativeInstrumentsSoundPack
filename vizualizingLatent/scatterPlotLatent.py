import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data=np.load('./../data/latentDataset.npz')



X_mu=data['X'][:500,:,0]
X_std=data['X'][:500,:,1]

y=data['y'][:500]

print(X_mu.shape,X_std.shape,y.shape)

target_ids = range(2)
target_names='regular drum','drum fill'
print(type(target_names))

target_ids = range(2)
target_names='regular drum','drum fill'



plt.figure(figsize=(6, 5))
colors = 'r', 'g'






tsne = TSNE(n_components=2, random_state=0)
X_mu_2d = tsne.fit_transform(X_mu)
X_std_2d=tsne.fit_transform(X_std)


# for i, c, label in zip(target_ids, colors,target_names ):
#     plt.scatter(X_std_2d[y == i, 0], X_std_2d[y == i, 1], c=c, label=label)
# plt.legend()
# plt.show()
print(X_mu_2d[y==0,0].shape)

for j in range(0,32):
    for w in range(j,32):

        for i, c, label in zip(target_ids, colors,target_names ):
            plt.scatter(X_mu[y == i,j], X_mu[y == i,w], c=c, label=label)
        plt.legend()
        plt.show()

