from classifier.utils.Metadata import Metadata
import numpy as np
from sklearn.externals import joblib

def predict(array):
    clf = joblib.load('./classifier/utils/models/warmness.pkl')

    scaler = joblib.load('./classifier/utils/models/scaler.pkl')
    array=array.reshape((1,array.shape[0],array.shape[1]))
    data=Metadata(array).metadata
    data['vae_embeddings'] = data['vae_embeddings'][:, 0:32]
    list_label = ['vae_embeddings','count','drums_pitches_used']
    list_x = []
    for label in list_label:
        list_x.append(data[label])
    X = np.concatenate(list_x, axis=1)
    X_std = scaler.transform(X)
    y = clf.predict(X_std)[0]






    return y

if __name__=='__main__':

    array=np.ones((96,128))
    proba=predict(array)
    print(proba)