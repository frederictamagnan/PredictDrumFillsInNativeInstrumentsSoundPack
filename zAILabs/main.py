
from classifier.classifier import predict
from sketchRnn.generateNewFill import generate

prefix='sketchrnn_'
sufix='_cleaned.pt'

methods=['Clustering_','RuleB_','Supervised_']
beta=['0.01','0.1','1','100','100','250']

#MODEL CHOICE
model_name=prefix+methods[2]+beta[3]+sufix

#GENERATION
fill=generate("./temp/test.mid",model_name=model_name)

#PREDICTION FOR INPUT 96*128
print(predict(fill))


#PREDICTION FOR INPUT 96*9
import numpy as np
fill2=np.zeros((96,9))
print(predict(fill2,reduced=True))

fill2=np.ones((96,9))
print(predict(fill2,reduced=True))


