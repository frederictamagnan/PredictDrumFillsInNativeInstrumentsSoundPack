import os
from pypianoroll import Multitrack,Track
import pypianoroll as ppr


path="/home/ftamagna/Documents/_AcademiaSinica/dataset/magentaDrums/"

# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print ('Error: Creating directory. ' + directory)
#
#
# for i,filename in enumerate(os.listdir(path)):
#     multi=ppr.parse(path+filename)
#     createFolder(path+str(i))
#     multi.save(path+str(i)+'/'+filename[:-4]+'.npz')

for i,filename in enumerate(os.listdir(path)):
    print(filename)


