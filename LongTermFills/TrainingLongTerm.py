import numpy as np
from LongTermNet import Encoder,DecoderFills,LongTermNet
from LongTermNetDataset import LongTermNetDataset
import torch
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class TrainingLongTerm:



    def __init__(self,batch_size,lr,n_epochs,dataset_filepath,beta,gru_hidden_size,linear_hidden_size,seq_len,num_directions,gru_embedding_hidden_size):

        self.dataset_filepath=dataset_filepath
        self.batch_size=batch_size
        self.lr=lr
        self.n_epochs=n_epochs
        self.beta=beta
        self.seq_len=seq_len
        self.num_directions=num_directions
        self.gru_hidden_size=gru_hidden_size
        self.linear_hidden_size=linear_hidden_size
        self.gru_embedding_hidden_size=gru_embedding_hidden_size

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')
        self.device=torch.device("cpu")


    def load_data(self,binarize=True):
        data = np.load(self.dataset_filepath)
        data=dict(data)
        track_array=data['track_array']
        self.dataset=LongTermNetDataset(numpy_array=track_array,use_cuda=self.use_cuda)
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


        self.train_loader = torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True,drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False,drop_last=True)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size,
                                                             shuffle=False,drop_last=True)

    def count_parameters(self,model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        print(params,"PARAMETERS_RNN")

    def train_model(self):

        self.loss_train_metrics=np.zeros((1,5))



        encoderFills= Encoder(num_features=9,gru_hidden_size=self.gru_hidden_size,gru_hidden_size_2=self.gru_hidden_size,seq_len=self.seq_len,num_directions=self.num_directions,linear_hidden_size=self.linear_hidden_size).to(self.device)
        encoderRegular= Encoder(num_features=9,gru_hidden_size=self.gru_hidden_size,gru_hidden_size_2=self.gru_hidden_size,seq_len=self.seq_len,num_directions=self.num_directions,linear_hidden_size=self.linear_hidden_size).to(self.device)
        decoderFills=DecoderFills(linear_hidden_size=self.linear_hidden_size,gru_embedding_hidden_size=self.gru_embedding_hidden_size).to(self.device)
        longTermNet=LongTermNet(encoderFills, decoderFills,encoderRegular)

        optimizer = optim.Adam(longTermNet.parameters(), lr=self.lr)
        test_err = 0

        for epoch in range(self.n_epochs):
            loss_sum = 0
            bce_sum = 0
            kld_f_sum = 0
            kld_r_sum = 0

            for batch_i, data in enumerate(self.train_loader):
                x, y = data
                # print(x.size(),y.size(),"X Y SIZE")
                optimizer.zero_grad()
                data_out = longTermNet(x,y)
                # print(data_out.size(),"dATA OUT")

                loss,bce,kld_f,kld_r=longTermNet.elbo(data_out,y)


                loss.backward()
                optimizer.step()

                loss_sum += loss.data.item()
                bce_sum += bce.data.item()
                kld_r_sum += kld_r.data.item()
                kld_f_sum += kld_f.data.item()


                if batch_i % 5 == 0:
                    print('Train Epoch: {} [{:4d}/{} ({:2.0f}%)]      Loss: {:.6f}'.format(
                        epoch,
                        batch_i * self.batch_size,
                        len(self.train_loader.dataset),
                        100. * batch_i / len(self.train_loader),
                        loss.data.item() / self.batch_size))
                    print('bce: {:.6f}, kld_f: {:.6f}, kld_r: {:.6f}'.format(
                        bce.data.item() / self.batch_size,
                        kld_f.data.item() / self.batch_size,kld_r.data.item() / self.batch_size))
            print('====> Epoch: {} Average loss: {:.4f}, bce: {:.4f}, kld_f: {:.4f},kld_r: {:.4f}'.format(
                epoch, loss_sum / len(self.train_loader.dataset),
                       bce_sum / len(self.train_loader.dataset),
                       kld_f_sum / len(self.train_loader.dataset),kld_r_sum / len(self.train_loader.dataset),
            ))
            #     print('Average bce: {:.4f}, kld: {:.4f}'.format(
            #         bce_sum / len(train_loader.dataset),
            #         kld_sum / len(train_loader.dataset)))



            # if epoch % 5 == 0:
            loss_sum_test = 0
            for batch_i, data in enumerate(self.test_loader):
                with torch.no_grad():
                    # x,y = Variable(data).type(torch.float32).to(self.device)
                    x,y=data
                    data_out = longTermNet(x,y)

                    loss = F.binary_cross_entropy(
                        data_out,
                        y,
                        reduction='sum'
                    )
                    loss_sum_test += loss.item()

            print('====> Testing Average Loss: {}'.format(
                loss_sum_test / len(self.test_loader.dataset)))

            row=np.asarray([epoch, loss_sum / len(self.train_loader.dataset),
             bce_sum / len(self.train_loader.dataset),
             kld_r_sum / len(self.train_loader.dataset),kld_f_sum / len(self.train_loader.dataset), loss_sum_test / len(self.test_loader.dataset)]).reshape((1,6))


            # self.loss_train_metrics = np.concatenate((self.loss_train_metrics, row))


        self.net=LongTermNet


    def save_model(self, filepath, name):
        torch.save(self.net.state_dict(), filepath + name)
        np.save('./metrics_training_'+name,self.loss_train_metrics)
        np.savez('./indices_'+name,train=self.train.indices,test=self.test.indices,validation=self.validation.indices)


if __name__=="__main__":

    LR=0.01
    BATCH_SIZE=256
    N_EPOCHS=210

    server=False
    if not(server):
        local_dataset = '/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/FillsExtractedFour.npz'
        tg=TrainingLongTerm(lr=LR,batch_size=BATCH_SIZE,n_epochs=N_EPOCHS,dataset_filepath=local_dataset,beta=0.5,linear_hidden_size=[64,32],gru_hidden_size=64,seq_len=16,num_directions=2,gru_embedding_hidden_size=16)
        tg.load_data()
        tg.split_data()
        tg.train_model()
        # tg.save_model("./../models/",'sketchrnn.pt')