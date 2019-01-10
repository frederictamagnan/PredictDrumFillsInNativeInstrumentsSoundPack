import numpy as np

rootdir='/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/MetricsFiles/_NImetrics_training.npz'

metricsnpz=np.load(rootdir)
for key in metricsnpz:
    print(key)
fills=metricsnpz['fills']
print(fills.shape)
print(fills[:,1].sum())
vae=metricsnpz['offbeat_notes']
print("lol",vae[100])