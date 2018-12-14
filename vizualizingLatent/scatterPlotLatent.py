import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data=np.load('./../data/latentDataset_fills_without_genre_2z.npz')



X_mu=data['X'][500:1000,:,0]
X_std=data['X'][500:1000,:,1]



y=data['y'][500:1000]


#test

# X_mu=np.ones((500,2,1))
# X_mu[:250]=np.zeros((2,1))
# print(X_mu)
#
# y=np.ones(500)
# y[:250]=0
# print(X_mu.shape)
print(X_mu.shape,X_std.shape,y.shape)

target_ids = range(2)
target_names='regular drum','drum fill'
print(type(target_names))

target_ids = range(2)
target_names='regular drum','drum fill'



# plt.figure(figsize=(5, 6))
colors = 'r', 'g'






# tsne = TSNE(n_components=2, random_state=0)
# X_mu_2d = tsne.fit_transform(X_mu)
# X_std_2d=tsne.fit_transform(X_std)

# print(X_mu_2d.shape,"shape")


# for i, c, label in zip(target_ids, colors,target_names ):
#     plt.scatter(X_mu[y == i, 0], X_mu[y == i, 1], c=c, label=label)
# plt.legend()
# plt.show()
# print(X_mu_2d[y==1,0].shape)


for i, c, label in zip(target_ids, colors,target_names ):
    plt.scatter(X_mu[y == i,0], X_mu[y == i,1], c=c, label=label)
plt.legend()
plt.show()

