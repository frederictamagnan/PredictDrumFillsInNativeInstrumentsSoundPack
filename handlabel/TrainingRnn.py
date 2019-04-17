import numpy as np
from RnnDataset import RnnDataset
from RnnNet import RnnNet
import torch
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn.functional as F
from utils import tensor_to_numpy
import torch.nn as nn
from torch.autograd import Variable

class TrainingRnn:



    def __init__(self,batch_size,lr,n_epochs,dataset_filepath):

        self.dataset_filepath=dataset_filepath
        self.batch_size=batch_size
        self.lr=lr
        self.n_epochs=n_epochs
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')



    def load_data(self,binarize=True):
        data = np.load(self.dataset_filepath)
        data=dict(data)
        X=data['X']
        y = data['y']
        print(y.sum(),"YYY")
        self.dataset=RnnDataset(X=X,y=y,use_cuda=self.use_cuda)
        print(len(self.dataset),"LEN DATASET",self.dataset.y.sum())


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


        self.train_loader = torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True,drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=True,drop_last=True)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.validation, batch_size=self.batch_size,
                                                             shuffle=True,drop_last=True)

    def count_parameters(self,model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params,"PARAMETERS_RNN")

    def train_model(self):
        rnn=RnnNet()
        self.net = rnn
        self.count_parameters(self.net)
        self.loss_train_metrics=np.zeros((1,5))

        optimizer = optim.Adam(rnn.parameters(), lr=self.lr)
        test_err = 0

        for epoch in range(self.n_epochs):
            loss_sum = 0

            for batch_i, data in enumerate(self.train_loader):
                x, y = data
                optimizer.zero_grad()
                data_out = rnn(x)

                # print(type(data))
                loss = F.binary_cross_entropy(data_out,y,reduction='sum')
                loss.backward()
                optimizer.step()

                loss_sum += loss.data.item()


                if batch_i % 5 == 0:
                    print('Train Epoch: {} [{:4d}/{} ({:2.0f}%)]      Loss: {:.6f}'.format(
                        epoch,
                        batch_i * self.batch_size,
                        len(self.train_loader.dataset),
                        100. * batch_i / len(self.train_loader),
                        loss.data.item() / self.batch_size))


            # if epoch % 5 == 0:
            loss_sum_test = 0
            for batch_i, data in enumerate(self.validation_loader):
                with torch.no_grad():
                    # data = Variable(data[0]).type(torch.float32).to(self.device)
                    x, y = data
                    # print(y,"lool y ")
                    data_out = rnn(x)

                    loss = F.binary_cross_entropy(data_out, y, reduction='sum')

                    loss_sum_test += loss.item()
                    self.accuracy(data_out, y), "accuracy"
            print('====> Testing Average Loss: {}'.format(
                loss_sum_test / len(self.validation_loader.dataset)))










    def save_model(self, filepath, name):
        torch.save(self.net.state_dict(), filepath + name)
        np.save('./metrics_training_'+name,self.loss_train_metrics)
        np.savez('./indices_'+name,train=self.train.indices,test=self.test.indices,validation=self.validation.indices)

    def accuracy(self,outputs,labels):

        o = tensor_to_numpy(outputs)

        # print(o.shape, "lool")
        l = tensor_to_numpy(labels)
        o = o.reshape(-1)
        l = l.reshape(-1)
        # print(o, l)
        o = (o > 0.5) * 1
        # print(o.shape, l.shape)
        acc = ((o == l) * 1).sum() / (o.shape[0])
        print("ACC", acc *100,"% ",o.sum(),l.sum())


if __name__=="__main__":

    LR=0.01
    BATCH_SIZE=100
    N_EPOCHS=20

    server=False
    if not(server):
        local_dataset = '/home/ftamagna/Documents/_AcademiaSinica/dataset/drumDetection/dataset.npz'

        tg=TrainingRnn(lr=LR,batch_size=BATCH_SIZE,n_epochs=N_EPOCHS,dataset_filepath=local_dataset)
        tg.load_data()
        tg.split_data()
        tg.train_model()
        tg.save_model("./../models/",'rnndetection.pt')




