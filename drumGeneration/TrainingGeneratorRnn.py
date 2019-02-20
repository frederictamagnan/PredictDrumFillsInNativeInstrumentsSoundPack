import numpy as np
from RnnGenerateDataset import RnnGenerateDataset
import torch
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn.functional as F
from RnnGenerateNet import RnnGenerateNet
import torch.nn as nn

class TrainingGenerator:



    def __init__(self,batch_size,lr,n_epochs,dataset_filepath):

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
        track_array=data['track_array']
        self.dataset=RnnGenerateDataset(numpy_array=track_array,use_cuda=self.use_cuda)
        print(len(self.dataset),"LEN DATASET")


    def split_data(self):
        ratio = 0.6
        train_length = int(len(self.dataset) * ratio)
        validation_length = int((len(self.dataset) - train_length) * 0.5)
        test_length = len(self.dataset) - validation_length - train_length
        self.train, self.validation, self.test = random_split(self.dataset,
                                                              [train_length, validation_length, test_length])

        print(len(self.train),"SELF TRAIN")
        print(len(self.validation), "SELF validation")
        print(len(self.test), "SELF test")


        self.train_loader = torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True,drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False,drop_last=False)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size,
                                                             shuffle=False,drop_last=False)

    def count_parameters(self,model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params,"PARAMETERS_RNN")

    def train_model(self):
        if self.use_cuda:
            rnn=RnnGenerateNet(batch_size=self.batch_size).cuda()
        else:
            rnn = RnnGenerateNet(batch_size=self.batch_size)

        criterion =  nn.BCELoss()
        optimizer = optim.SGD(rnn.parameters(), lr=self.lr, momentum=0.9,weight_decay=0.6)
        self.count_parameters(rnn)

        for epoch in range(self.n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = rnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            with torch.no_grad():
                total_val_loss = 0
                for j, data in enumerate(self.validation_loader):
                    inputs, labels = data
                    val_outputs = rnn(inputs)
                    val_loss_size = criterion(val_outputs, labels)
                    total_val_loss += val_loss_size.data[0]
                print("epochs ",str(epoch)," : val loss",total_val_loss / (j+1), "training loss", running_loss / (i+1))

        print('Finished Training')

        self.net = rnn

    def save_model(self, filepath, name):
        torch.save(self.net.state_dict(), filepath + name)

if __name__=="__main__":

    LR=0.001
    BATCH_SIZE=2048
    N_EPOCHS=200

    server=False
    if not(server):
        local_dataset = '/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/bigsupervised.npz'

        tg=TrainingGenerator(lr=LR,batch_size=BATCH_SIZE,n_epochs=N_EPOCHS,dataset_filepath=local_dataset)
        tg.load_data()
        tg.split_data()
        tg.train_model()
        tg.save_model("./../models/",'vae_generation.pt')




