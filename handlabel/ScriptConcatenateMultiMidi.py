filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumDetection/good/'
import pypianoroll as ppr
from pypianoroll import Multitrack,Track
import os
l=os.listdir(filepath)
from DrumReducerExpander import DrumReducerExpander
import numpy as np
enc=DrumReducerExpander()

data=np.zeros((1,3,16,9))
for elt in l:
    # print(elt)
    multi=ppr.parse(filepath+elt)
    drum_track=multi.tracks[0].pianoroll
    drum_reduced=enc.encode(drum_track,no_batch=True)
    drum_reduced=enc.encode_808(drum_reduced,no_batch=True)
    drum_reduced=drum_reduced[16:-16,:]
    # print(drum_reduced.shape)
    if drum_reduced.shape[0]==48:
        drum_reduced=drum_reduced.reshape((1,3,16,9))
        data=np.concatenate((data,drum_reduced))

filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumDetection/bad/'
l=os.listdir(filepath)

data_bad=np.zeros((1,3,16,9))
for elt in l:
    # print(elt)
    multi=ppr.parse(filepath+elt)
    drum_track=multi.tracks[0].pianoroll
    drum_reduced=enc.encode(drum_track,no_batch=True)
    drum_reduced=enc.encode_808(drum_reduced,no_batch=True)
    drum_reduced=drum_reduced[16:-16,:]
    # print(drum_reduced.shape)
    if drum_reduced.shape[0]==48:
        drum_reduced=drum_reduced.reshape((1,3,16,9))
        data_bad=np.concatenate((data_bad,drum_reduced))

data2=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumDetection/'+'dataset2.npz')
data2=dict(data2)
data2=data2['track_array']

y=np.concatenate((np.ones(data.shape[0]),np.zeros(data2.shape[0]+data_bad.shape[0])))

data_=np.concatenate((data,data2,data_bad))

np.savez(filepath+"dataset",X=data_,y=y)

