from Generator_vae import Generator
import numpy as np
server=False

if server:
    model_path = '/home/ftamagnan/PredictDrumFillsInNativeInstrumentsSoundPack/models/'
    model_name = 'generation_model.pt'

else:
    model_path='/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/models/'
    model_name = 'vae_generation_cleaned_v2.pt'

    dataset_path='/home/ftamagna/Documents/_AcademiaSinica/dataset/lpd_5/lpd_5_cleansed/'
    tags_path= ['/home/ftamagna/Documents/_AcademiaSinica/code/LabelDrumFills/id_lists/tagtraum/tagtraum_Rock.id']
    temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
    indices_path='lol'





#train validation creation from indices


for i_name,name in enumerate(['Supervised']):


    # for beta in [0.01,0.1,1,100,250]:
    for i_beta,beta in enumerate([ 250]):
        model_name='sketchrnn_'+name+'_'+str(beta)+'_kld.pt'
        temp_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/temp/'
        dataset_path = '/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtracted' + name + '_cleaned.npz'
        indices_path = '/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/indices/indices_sketchrnn_' + name + '_' + str(
            beta) + '_kld.pt.npz'
        model_name = 'vae_generation__c_cleaned_v2.pt'

        # dataset = dict(np.load(dataset_path))
        # indices = dict(np.load(indices_path))
        # x=100
        # offset=30
        # genre = dataset['genre'][indices['validation']][x:x+offset]
        # beat = dataset['track_array'][indices['validation']][x:x+offset]
        # dataset = dataset['vae'][indices['validation']][x:x+offset]
        # print(genre.shape,beat.shape,dataset.shape,"before")
        d = np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/arr.npz')
        beat = dict(d)['track_array']
        dataset=dict(d)['vae']
        dataset=dataset.reshape(dataset.shape[0],2,32,2)
        # beat = array.reshape(array.shape[0], 1, array.shape[1], array.shape[2])
        genre=dict(d)['genre']
        genre=genre.reshape((genre.shape[0],genre.shape[1],1))
        print(genre.shape,beat.shape,dataset.shape,"after")



        g=Generator(model_path=model_path,model_name=model_name,dataset_path=dataset_path,tags_path=tags_path,temp_filepath=temp_filepath)

        g.generate_from_magenta(dataset,genre,beat=beat,tag="_method_2",th=0.30)

