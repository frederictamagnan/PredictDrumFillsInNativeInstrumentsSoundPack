

import numpy as np
from torch.utils.data.dataset import random_split
from utils import tensor_to_numpy
import json
from DnnDataset import DnnDataset
from DnnNet import DnnNet
from WideAndDeepDataset import WideAndDeepDataset
from WideAndDeepNet import WideAndDeepNet
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch
from sklearn.metrics import confusion_matrix


class TrainingClassifier:


    def __init__(self,metrics_dir,metrics_filename,filepath_filename,batch_size,n_epochs,lr,downsampling=True,upsampling=False):
        # torch.manual_seed(1)
        self.metrics_dir=metrics_dir
        self.metrics_filename=metrics_filename
        self.filepath_filename=filepath_filename
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.lr=lr
        self.downsampling=downsampling
        self.upsampling=upsampling


    def load_dataset(self):
        self.data_raw=np.load(self.metrics_dir+self.metrics_filename)
        self.data_raw=dict(self.data_raw)

        with open(self.metrics_dir+self.filepath_filename) as data_file:
            self.list_filepath=json.load(data_file)


        self.dataset=DnnDataset(self.data_raw,self.list_filepath,downsampling=self.downsampling,upsampling=self.upsampling)


    # def load_dataset_wide(self):
    #     self.data_raw=np.load(self.metrics_dir+self.metrics_filename)
    #     self.data_raw=dict(self.data_raw)
    #
    #     with open(self.metrics_dir+self.filepath_filename) as data_file:
    #         self.list_filepath=json.load(data_file)
    #
    #
    #     self.dataset=WideAndDeepDataset(self.data_raw,self.list_filepath,downsampling=self.downsampling)

    def split_dataset(self):

        ratio=0.6
        train_length= int(len(self.dataset)* ratio)
        validation_length=int((len(self.dataset)-train_length)*0.5)
        test_length=len(self.dataset)-validation_length-train_length
        self.train,self.validation,self.test=random_split(self.dataset,[train_length,validation_length,test_length])

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=len(self.test), shuffle=False)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=len(self.validation), shuffle=False)


    def train_model(self):
        dnn=DnnNet(deepdata=self.dataset.X_deep)
        # optimizer = optim.Adam(dnn.parameters(), lr=self.lr)
        criterion = F.binary_cross_entropy
        optimizer = optim.Adam(dnn.parameters(), lr=0.001, betas=(0.9, 0.999))
        for epoch in range(self.n_epochs):

            for i, ( X_deep, target) in enumerate(self.train_loader):
                X_d = Variable(X_deep).float()
                y = Variable(target).float()

                optimizer.zero_grad()
                y_pred = dnn(X_d)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                if i % 1000 == 0:
                    print('Epoch {} of {}, Loss: {}'.format(epoch + 1, self.n_epochs, loss.data[0]))

            correct = 0
            total = 0
            # with torch.no_grad():
            for i, (X_deep, target) in enumerate(self.validation_loader):
                X_d = Variable(X_deep).float()
                y = Variable(target).float()
                y_pred = dnn( X_d)
                y_pred_cat = (y_pred > 0.5).squeeze(1).float()

                total += y.size(0)
                correct+= (y_pred_cat.squeeze() == y.squeeze()).sum().item()
                if i % 1000 == 0:
                    print('Epoch {} of {}, ValLoss: {}'.format(epoch + 1, self.n_epochs, loss.data[0]))

            print('Validation accuracy: %d %%' % (
                100 * correct / total))

        y_pred_total = []
        y_true_total = []

        correct = 0
        total = 0

        with torch.no_grad():
            for i, (X_deep, target) in enumerate(self.test_loader):
                X_d = Variable(X_deep).float()
                y = Variable(target).float()
                y_pred = dnn(X_d)
                y_pred_cat = (y_pred > 0.5).squeeze(1).float()

                y_pred_total.append(tensor_to_numpy(y_pred_cat).astype(int))
                y_true_total.append(tensor_to_numpy(y.squeeze(1).float()).astype(int))


                total += y.size(0)
                # correct += (y_pred_cat == y).sum().item()
                correct+= (y_pred_cat.squeeze() == y.squeeze()).sum().item()
                print(total,correct)
        print('test accuracy: %d %%' % (
                100 * correct / total))

        y_pred_total = np.concatenate(y_pred_total)
        y_true_total = np.concatenate(y_true_total)


        conf = confusion_matrix(y_true_total, y_pred_total).ravel()
        print(conf / total * 100), "Normalized value of tn fp fn tp"
        print(conf, "value of tn fp fn tp")
        tn, fp, fn, tp = conf

        print(tp / (fp + tp), "Precision")
        print(tp / (fn + tp), "Recall")
        model_parameters = filter(lambda p: p.requires_grad, dnn.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])


        print("NUMBER OF PARAMETER : " + str(params))





    # def train_model_wide(self):
    #     wnd = WideAndDeepNet(deepdata=self.dataset.X_deep,widedata=self.dataset.X_wide)
    #     optimizer = optim.Adam(wnd.parameters(), lr=self.lr)
    #     criterion = F.binary_cross_entropy
    #     for epoch in range(self.n_epochs):
    #
    #         for i, (X_wide,X_deep, target) in enumerate(self.train_loader):
    #             X_d = Variable(X_deep).float()
    #             X_w=Variable(X_wide).float()
    #             y = Variable(target).float()
    #
    #             optimizer.zero_grad()
    #             y_pred = wnd(X_w,X_d)
    #             loss = criterion(y_pred, y)
    #             loss.backward()
    #             optimizer.step()
    #             if i % 1000 == 0:
    #                 print('Epoch {} of {}, Loss: {}'.format(epoch + 1, self.n_epochs, loss.data[0]))
    #
    #         correct = 0
    #         total = 0
    #         # with torch.no_grad():
    #         for i, (X_wide,X_deep, target) in enumerate(self.validation_loader):
    #             X_d = Variable(X_deep).float()
    #             X_w = Variable(X_wide).float()
    #             y = Variable(target).float()
    #             y_pred = wnd(X_w,X_d)
    #             y_pred_cat = (y_pred > 0.5).squeeze(1).float()
    #
    #             total += y.size(0)
    #             correct += (y_pred_cat == y).sum().item()
    #
    #         print('Validation accuracy: %d %%' % (
    #                 100 * correct / total))
    #
    #     # with torch.no_grad():
    #     for i, (X_wide,X_deep, target) in enumerate(self.test_loader):
    #         X_d = Variable(X_deep).float()
    #         X_w = Variable(X_wide).float()
    #         y = Variable(target).float()
    #         y_pred = wnd(X_w,X_d)
    #         y_pred_cat = (y_pred > 0.5).squeeze(1).float()
    #
    #         total += y.size(0)
    #         correct += (y_pred_cat == y).sum().item()
    #
    #     print('test accuracy: %d %%' % (
    #             100 * correct / total))
    #
    #     model_parameters = filter(lambda p: p.requires_grad, wnd.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #
    #
    #     print("NUMBER OF PARAMETER : " + str(params))



if __name__=='__main__':

    metricsdir = '/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/MetricsFiles/'
    metricsfilename = 'total_metrics_training.npz'
    filepathfilename = 'total_list_filepath.json'

    BATCH_SIZE=512
    N_EPOCHS=100
    LR = 0.0001
    tc=TrainingClassifier(metricsdir,metricsfilename,filepathfilename,BATCH_SIZE,N_EPOCHS,LR,downsampling=True,upsampling=False)
    tc.load_dataset()
    # for i in range(1000):
    #     print(tc.dataset.Y[i],tc.dataset.list_filepath[i])

    tc.split_dataset()
    print(tc.dataset.Y.sum())
    print(len(tc.dataset.Y))
    tc.train_model()







