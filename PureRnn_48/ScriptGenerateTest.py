from GeneratorSketchRnn import GeneratorSketchRnn
import numpy as np
server=False
dataset_array='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/48/'
name_array='test.npy'

model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
model_name='rnn_48.pt'
temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=None,tags_path=None,temp_filepath=temp_filepath)
g.count_parameters()
# g.generate(10,save=False)
array = np.load(dataset_array + name_array)
array=array[0:60]

g.generate_from(array,tag="_method_48",th=0.4)

