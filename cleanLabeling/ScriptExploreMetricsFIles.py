import numpy as np

rootdir='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/MetricsFiles/total_metrics_training.npz'

metricsnpz=np.load(rootdir)
for key in metricsnpz:
    print(key)
    print(metricsnpz[key][100])
    print(metricsnpz[key].shape)
fills=metricsnpz['fills']
# print(fills.shape)
print(fills[:,1].sum())

#
#
# ROOTDIR='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/RawFiles/'
#
# rawfiles=np.load(ROOTDIR+"NI_raw.npz")
#
# print(rawfiles['X'].shape)