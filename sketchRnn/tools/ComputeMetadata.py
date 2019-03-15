ROOTDIR='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/RawFiles/'
savedir='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/MetadataFiles/'
import json
from MetadataTraining import MetadataTraining
import numpy as np
class ComputeMetadata:

    def __init__(self,rootdir,dataset='NI'):
        self.rootdir = rootdir
        self.list_file=[('NI_raw.npz','NI_filepath.json'),('OG_raw.npz','OG_filepath.json')]
        self.compute_metadata()



    def compute_metadata(self):
        lmetadata=[]
        for dataset in self.list_file:
            data_raw = np.load(self.rootdir+dataset[0])
            data_raw=data_raw['X']

            with open(self.rootdir+dataset[1]) as data_file:
                list_filepath = json.load(data_file)

            metadata=MetadataTraining(data_raw,list_filepath)
            metadata.save_metadata(savedir,dataset[0][:2])
            lmetadata.append(metadata)

        for key in lmetadata[0].metadata.keys():
            lmetadata[0].metadata[key]=np.concatenate((lmetadata[0].metadata[key],lmetadata[1].metadata[key]))
        lmetadata[0].metadata['list_filepath']=lmetadata[0].list_filepath+lmetadata[1].list_filepath
        lmetadata[0].save_metadata(savedir, 'total')


if __name__=='__main__':

    cm=ComputeMetadata(ROOTDIR,savedir)







