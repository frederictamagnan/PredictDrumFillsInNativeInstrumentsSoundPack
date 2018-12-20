import os
import time
import pypianoroll
from pypianoroll import Multitrack, Track
import torch.utils.data as Data
from matplotlib import pyplot as plt
# from models.vae_rnn import *
from utils import parse_data
from DrumsDataset import DrumsDataset
from utils import tensor_to_numpy
from models.vae_rnn_batch_size import *
dataset_path='./../data/'


# model3='vae_L1E-02_beta2E+01_beat48_loss4E+01_tanh_gru32_e20_b256_hd16-2_20181212_105409.pt'


# vae.load_state_dict(torch.load("./../models/vae_L1E-02_beta2E+01_beat48_loss2E+01_tanh_gru32_e100_b256_hd64-32_20181211_134833.pt",map_location='cpu'))
# vae.load_state_dict(torch.load("./../models/"+model3,map_location="cpu"))


data = np.load(dataset_path+'dataset_genres_fill.npz')
X=data['X']
y=data['y']
BATCH_SIZE=123

encoder = Encoder(batch_size=BATCH_SIZE).to(device)
decoder = Decoder(beat=48,batch_size=BATCH_SIZE).to(device)
vae = VAE(encoder, decoder).to(device)
vae.load_state_dict(torch.load("./../models/vae_L1E-02_beta2E+01_beat48_loss2E+01_tanh_gru32_e100_b256_hd64-32_20181008_034323.pt",map_location='cpu'))

print(y.shape,"y shape")
# w=y[:,1,:]

# print(w.shape)
# print(w[:300])
# index_fills=np.where(w[:,0]==1)

# train_x_reduced=X[index_fills[0],:,:]

train_x_reduced=X

print(train_x_reduced.shape)



TESTING_RATIO = 0.1
N_DATA = train_x_reduced.shape[0]
N_TRAINING = int(train_x_reduced.shape[0]*TESTING_RATIO)
N_TESTING = N_DATA - N_TRAINING
print(N_TRAINING,"n training")
train_x, test_x = parse_data(train_x_reduced,TESTING_RATIO)
# train_dataset = Data.TensorDataset(train_x)
# test_dataset = Data.TensorDataset(test_x)

print(X.shape,train_x.shape)
train_dataset = DrumsDataset(X)
test_dataset = DrumsDataset(test_x)

train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=1,
)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)

#warning
global_tensor=np.zeros((1,32,2))
global_labels=np.zeros((1,2,1))

for batch_i, (data,index) in enumerate(train_loader):
    with torch.no_grad():
        # data is a one element array
        # data = data[0,:]

        data = Variable(data).type(torch.float32).to(device)
        latent_mu = vae._enc_mu(encoder(data))
        latent_std=torch.exp(vae._enc_log_sigma(encoder(data)))
        print(latent_std.shape,latent_mu.shape,"SHAPE LATENT")
        # the distance between first 2 latent vectors

        indexes=tensor_to_numpy(index)

        latent_mu=tensor_to_numpy(latent_mu)
        latent_std=tensor_to_numpy(latent_std)


        latent_tensor=np.stack((latent_mu,latent_std),axis=2)

        labels=y[indexes,:,:]
        print(labels.shape,"label shape")
        global_labels=np.concatenate((global_labels,labels))

        global_tensor=np.concatenate((global_tensor,latent_tensor))





        print(batch_i)

print(global_labels.shape,"labels shape")
print(global_tensor.shape,"global tensor shape")


global_labels=global_labels[1:]
global_tensor=global_tensor[1:,:,:]


print(global_labels.sum())

# np.savez('./../data/latentDataset_genre_fill_batchsize.npz',X=global_tensor,y=global_labels)

