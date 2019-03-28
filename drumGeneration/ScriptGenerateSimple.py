import numpy as np
from DrumReducerExpander import DrumReducerExpander
from pypianoroll import Multitrack
from utils import random_file_genre
# array_vae=np.zeros((1,3,64))
array_vae=np.zeros((1,2,64))

genre=np.zeros((1,16,1))
# X=np.zeros((1,288,128))
X=np.zeros((1,192,128))

list_filepath=[]
i=0
n=100
while i<n:
    rf=random_file_genre()
    # print(rf)
    list_filepath.append(rf)
    multi=Multitrack(rf[0]+rf[1])
    multi.binarize()

    piano=multi.tracks[0].pianoroll
    length=len(piano)
    mid=int(length//96//2)
    # x=piano[mid*96:(mid+3)*96]
    x=piano[mid*96:(mid+2)*96]

    x=x.reshape((1,x.shape[0],x.shape[1]))
    g=np.zeros((1,16,1))
    # print(rf,"RF")
    g[:,rf[2],:]=1
    try:
        metrics = dict(np.load(rf[0] + rf[1].replace('.npz', '_metadata_training.npz')))

        vae = metrics['vae_embeddings']
        # vae = vae[mid:(mid + 3)].reshape((-1, 3, 64))
        vae = vae[mid:(mid + 2)].reshape((-1, 2, 64))
        print(x.shape,X.shape)
        X=np.concatenate((X,x))
        print("open")

        genre=np.concatenate((genre,g))
        array_vae=np.concatenate((array_vae,vae))
        i+=1
        # print(i,'INDEXXXXXXXXX')
    except:
        print("error")


genre=genre[1:]
X=X[1:]
array_vae=array_vae[1:]
enc=DrumReducerExpander()
X=enc.encode(X)
X=enc.encode_808(X)
X=X.reshape(X.shape[0],2,16,9)
np.savez('/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/arr.npz',track_array=X,genre=genre,vae=array_vae)