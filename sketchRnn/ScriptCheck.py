import pypianoroll as ppr
import numpy as np
from pypianoroll import Track,Multitrack
from DrumReducerExpander import DrumReducerExpander
tr=Multitrack('/home/ftamagna/Documents/_AcademiaSinica/dataset/SKETCHBON/highLow.mid')

enc=DrumReducerExpander()

a=tr.tracks[0].pianoroll
a=enc.encode(a,no_batch=True)
a=enc.encode_808(a,no_batch=True)
np.savetxt('./lol.txt',a,fmt='%.0e')
print(a)