from pypianoroll import Track,Multitrack
import os
import sys



genre=['Pop','Funk','Jazz','Hard Rock','Metal','Blues & Country','Blues Rock','Ballad','Indie Rock','Indie Disco','Punk Rock']

selected_genre='Rock'


import pypianoroll as ppr
rootdir="/home/ftamagna/Documents/_AcademiaSinica/dataset/NI_Drum_Studio_Midi_encoded/MIDI Files"
temp="/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/"

counter_var=0
counter_var_96=0
for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if "Variation" in file:
                filepath=os.path.join(subdir,file)
                # print(filepath)
                multi=Multitrack(filepath)
                multi.binarize()
                print(multi.beat_resolution)
                if multi.tracks[0].pianoroll.shape[0]==96:
                    counter_var_96+=1
                counter_var+=1
                print(multi.tracks[0].pianoroll.shape)
                print(multi.tracks[0].pianoroll.sum())
                # ppr.write(multi,temp+"temp.mid")
                sys.exit()

print(counter_var_96,counter_var)