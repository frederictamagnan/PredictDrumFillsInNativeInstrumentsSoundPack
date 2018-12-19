

import midi

def changeChannel(rootdir,newdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file not in [".DS_Store"]:
                filepath=os.path.join(subdir, file)

                pattern = midi.read_midifile(filepath)

                for i,elt in enumerate(pattern[0]):
                    try:
                        elt.channel=9

                    except:
                        print("lol")

                newfilepath=newdir+filepath[len(rootdir):-(len(file))]
                if not os.path.exists(newfilepath):
                    os.makedirs(newfilepath)
                midi.write_midifile(newfilepath+'/'+file, pattern)



rootdir="/home/ftamagna/Documents/_AcademiaSinica/dataset/NI_Drum_Studio_Midi/MIDI Files"
import os

newdir="/home/ftamagna/Documents/_AcademiaSinica/dataset//NI_Drum_Studio_Midi_encoded/MIDI Files"


changeChannel(rootdir,newdir)








