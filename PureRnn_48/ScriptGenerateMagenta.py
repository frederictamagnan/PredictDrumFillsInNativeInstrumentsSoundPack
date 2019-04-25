from GeneratorSketchRnn import GeneratorSketchRnn
import numpy as np
server=False
dataset_array='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'
name_array='validation.npz'
if server:
    model_path = '/home/ftamagnan/PredictDrumFillsInNativeInstrumentsSoundPack/models/'
    model_name = 'generation_model.pt'

else:
    model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
    model_name = 'sketchrnn.pt'

    dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'
    tags_path= ['/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id']
    temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
    indices_path='lol'
th=[0.23,0.21,0.23]
for i_name,name in enumerate(['Supervised',"Supervised03",'RuleBased']):

    model_name='sketchrnn_'+name+'_v4.pt'
    temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
    g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
    g.count_parameters()
    # g.generate(10,save=False)
    array = np.load(dataset_array + name_array)

    array=dict(array)
    array=array['track_array'][0:30]
    array=array[:,0,:,:].reshape((array.shape[0],1,16,9))
    g.generate_from_magenta(array,tag="_method_"+str(i_name),th=th[i_name])

