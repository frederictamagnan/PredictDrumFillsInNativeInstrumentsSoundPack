from utils import random_file_genre
from pypianoroll import Multitrack,Track
import pypianoroll as ppr
from DrumReducerExpander import DrumReducerExpander
import numpy as np
import random
random.seed(24)
temp_filepath = '/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
n=20
multi_list=[]
new_list=[]
enc_dec=DrumReducerExpander()

vertical = False


for j in range(10):
    sum_pianoroll=np.zeros((96,9))
    for i in range(3):
        again=True
        while again:
            rf = random_file_genre()
            multi=Multitrack(rf[0]+rf[1])
            if 0 not in multi.get_empty_tracks():
                again=False
        print(len(multi.tracks[0].pianoroll),"LEN PIANOROLL")


        ppr.write(multi,filepath=temp_filepath+str(j)+"_"+str(i))
        pianoroll=multi.tracks[0].pianoroll[n*96:(n+1)*96,:]
        pianoroll_reduced=enc_dec.encode(pianoroll,no_batch=True)
        new_pianoroll=np.zeros(pianoroll_reduced.shape)
        if vertical:
            new_pianoroll[:,i*3]=pianoroll_reduced[:,i*3]
            new_pianoroll[:,(i+1)*3-1]=pianoroll_reduced[:,(i+1)*3-1]

        else:
            new_pianoroll[i*32:(i+1)*32]=pianoroll_reduced[i*32:(i+1)*32]





        sum_pianoroll+=new_pianoroll

    try:
        pianoroll_reduced=sum_pianoroll
        print(pianoroll_reduced.shape)

        pianoroll=enc_dec.decode(pianoroll_reduced,no_batch=True)
        pianoroll=np.concatenate((pianoroll,pianoroll),axis=0)
        print(pianoroll.shape,"shape pianoroll reapted")
        track=Track(pianoroll,is_drum=True)
        multi=Multitrack(tracks=[track])
        ppr.write(multi,temp_filepath+str(j)+"new_track")
    except:
        print("error shape")








