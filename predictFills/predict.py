import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
data=np.load('./../data/latentDataset_genre_fill.npz')

h = .02
X_mu_raw=data['X'][:,:,0]
X_std_taw=data['X'][:,:,1]



y_raw=data['y']
y_bis=y_raw[:,0,0].reshape((-1,1))
print(X_mu_raw.shape)
print(y_bis.shape)
X=np.concatenate((X_mu_raw,y_bis),axis=1)
y=y_raw[:,1,0]


X_0=X[y==0]
y_0=y[y==0]

X_1=X[y==1]
y_1=y[y==1]

X=np.concatenate((X_0[:X_1.shape[0]],X_1))
y=np.concatenate((y_0[:y_1.shape[0]],y_1))





print(y.sum()/len(y))
accuracy=[]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)

x=[]

# metrics=['uniform', 'distance']
metrics=['distance']
for n_neighbors in range(2,6):

    for weights in metrics:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        accuracy.append(accuracy_score(y_pred,y_test))
        x.append(n_neighbors)


print(accuracy)
print(len(x),len(accuracy))
plt.plot(x,accuracy)
plt.show()


# x=[]
# accuracy=[]
# for max_features in tqdm(range(10,25)):
#     # we create an instance of Neighbours Classifier and fit the data.
#     clf = RandomForestClassifier(n_estimators=160, max_depth=11,random_state=0,max_features=max_features)
#     clf.fit(X_train, y_train)
#     y_pred=clf.predict(X_test)
#     accuracy.append(accuracy_score(y_pred,y_test))
#     x.append(max_features)
#
#
# print(accuracy)
# print(len(x),len(accuracy))
# plt.plot(x,accuracy)
# plt.show()