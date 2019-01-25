import numpy as np

path="/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/MetricsFiles/"
name="total_metrics_training.npz"


data=dict(np.load(path+name))

list_x=[]
list_label=['vae_embeddings','offbeat_notes','drums_pitches_used','velocity_metrics']
for key in data.keys():
    if key in list_label:
        list_x.append(data[key])
        print(key)
X=np.concatenate(list_x,axis=1)
y=data['fills'][:,1].reshape(-1)


print(X.shape)
print(y.shape)


from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=2, random_state=0,
                           multi_class='ovr',penalty='l1',solver='saga',max_iter=3000).fit(X, y)
print(clf.predict(X[:2, :]))

print(clf.predict_proba(X[:2, :]).shape)

print(clf.score(X, y) )