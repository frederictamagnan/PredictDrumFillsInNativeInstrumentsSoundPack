import numpy as np

for name in ['diff','clustering','supervised']:
    a=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/metrics_training_sketchrnn_'+name+'.pt.npy')

    # a=a.reshape((-1,5))
    np.savetxt(name+'loss.csv',a,delimiter=',')