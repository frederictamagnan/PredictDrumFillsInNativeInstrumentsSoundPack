from GeneratorSketchRnn import GeneratorSketchRnn
import numpy as np
server=False
dataset_array='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/RawFiles/'
name_array='NI_raw.npz'
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
for i_name,name in enumerate(['Supervised']):


    # for beta in [0.01,0.1,1,100,250]:
    for i_beta,beta in enumerate([ 0.1,250]):
        model_name='sketchrnn_'+name+'_'+str(beta)+'_kld.pt'
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
        g=GeneratorSketchRnn(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)
        g.count_parameters()
        # g.generate(10,save=False)
        array=np.load(dataset_array+name_array)
        array=dict(array)
        array=array['X']
        np.random.seed(8)
        np.random.shuffle(array)
        array=array[200:300]
        g.generate_from(array,tag="_method_"+str(i_beta),th=0.3)
        # for i in range(10):
        #     g.generate_long(str(i)+"_method_"+str(i_beta),array[i])
