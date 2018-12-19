import os
ROOTDIR="/home/ftamagna/Documents/_AcademiaSinica/dataset/oddgrooves/ODDGROOVES_FILL_PACK/OddGrooves Fill Pack General MIDI"
from pathlib import Path
from pypianoroll import Multitrack


for filepath in Path(ROOTDIR).glob('./**/*'):
    if filepath.is_file():
        print(filepath)
        multi=Multitrack(str(filepath))
        print(multi.beat_resolution)
