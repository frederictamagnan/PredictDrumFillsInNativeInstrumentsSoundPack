ROOTDIR='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/RawFiles/'
savedir='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/MetricsFiles/'
import json
from MetricsTraining import MetricsTraining
import numpy as np
class ComputeMetrics:

    def __init__(self,rootdir,dataset='NI'):
        self.rootdir = rootdir
        self.list_file=[('NI_raw.npz','NI_filepath.json'),('OG_raw.npz','OG_filepath.json')]
        self.compute_metrics()



    def compute_metrics(self):
        lmetrics=[]
        for dataset in self.list_file:
            data_raw = np.load(self.rootdir+dataset[0])
            data_raw=data_raw['X']

            with open(self.rootdir+dataset[1]) as data_file:
                list_filepath = json.load(data_file)

            metrics=MetricsTraining(data_raw,list_filepath)
            metrics.save_metrics(savedir,dataset[0][:2])
            lmetrics.append(metrics)

        for key in lmetrics[0].metrics.keys():
            lmetrics[0].metrics[key]=np.concatenate((lmetrics[0].metrics[key],lmetrics[1].metrics[key]))
        lmetrics[0].metrics['list_filepath']=lmetrics[0].list_filepath+lmetrics[1].list_filepath
        lmetrics[0].save_metrics(savedir, 'total')


if __name__=='__main__':

    cm=ComputeMetrics(ROOTDIR,savedir)







