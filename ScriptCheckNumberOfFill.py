import os
rootdir="/home/ftamagna/Documents/_AcademiaSinica/dataset/NI_Drum_Studio_Midi/MIDI Files"

fill=0
count=0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print("lol")
        count+=1
        filepath=os.path.join(subdir,file)
        if "Fill" in filepath:
            fill+=1


print(fill,count)