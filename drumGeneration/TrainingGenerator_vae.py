import numpy as np
from DrumsDataset_vae_genre import DrumsDataset
import torch
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn as nn
from DNnet import DNnet

local_dataset='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtractedSupervised_cleaned.npz'

class TrainingGenerator:



    def __init__(self,batch_size,lr,n_epochs,dataset_filepath=local_dataset):

        self.dataset_filepath=dataset_filepath
        self.batch_size=batch_size
        self.lr=lr
        self.n_epochs=n_epochs

        self.use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')



    def load_data(self,binarize=True):
        data = np.load(self.dataset_filepath)
        data=dict(data)
        vae=data['vae']
        genre=data['genre']
        self.dataset=DrumsDataset(numpy_array=vae,genre=genre,use_cuda=self.use_cuda)
        print(len(self.dataset),"LEN DATASET")
    def split_data(self):
        ratio = 0.6
        train_length = int(len(self.dataset) * ratio)
        validation_length = int((len(self.dataset) - train_length) * 0.5)
        test_length = len(self.dataset) - validation_length - train_length
        self.train, self.validation, self.test = random_split(self.dataset,
                                                              [train_length, validation_length, test_length])

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True,drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False,drop_last=False)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size,
                                                             shuffle=False,drop_last=True)

    def train_model(self):
        if self.use_cuda:
            dnn=DNnet().cuda()
        else:
            dnn=DNnet()

        criterion = nn.MSELoss()
        optimizer = optim.SGD(dnn.parameters(), lr=self.lr, momentum=0.9,weight_decay=0.6)


        for epoch in range(self.n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs,g, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = dnn(inputs,g)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.10f' %
                          (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
            with torch.no_grad():
                total_val_loss = 0
                for i, data in enumerate(self.test_loader, 0):
                    inputs, g,labels = data
                    val_outputs = dnn(inputs,g)
                    val_loss_size = criterion(val_outputs, labels)
                    total_val_loss += val_loss_size.item()
                print('loss: %.10f' %(total_val_loss / (i+1)))
                print(len(self.test))

        print('Finished Training')

        self.net = dnn

    def save_model(self, filepath, name):
        torch.save(self.net.state_dict(), filepath + name)

if __name__=="__main__":

    LR=0.001
    BATCH_SIZE=4096*2
    N_EPOCHS=500

    tg=TrainingGenerator(lr=LR,batch_size=BATCH_SIZE,n_epochs=N_EPOCHS)
    tg.load_data()
    tg.split_data()
    tg.train_model()
    tg.save_model("./../models/",'vae_generation_cleaned.pt')