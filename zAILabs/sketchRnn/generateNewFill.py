from pypianoroll import Multitrack
import warnings
from sketchRnn.GeneratorSketchRnn import GeneratorSketchRnn
import pypianoroll as ppr

def generate(mid_path,model_name="sketch_rnn.pt"):


        multi=ppr.parse(mid_path)
        if multi.get_active_length()>96:
            warnings.warn("your midi file has a length >96 timesteps, only the first 96 timesteps will be taken ")

        if multi.beat_resolution != 24:
            raise " beat resolution != 24"


        array=multi.tracks[0].pianoroll[:96,:]
        array=(array>0)*1

        g = GeneratorSketchRnn(model_name=model_name)
        return g.generate(array)