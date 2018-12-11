import os
import time
import pypianoroll
from pypianoroll import Multitrack, Track
import torch.utils.data as Data
from matplotlib import pyplot as plt
from models.vae_rnn import *
from utils import parse_data


dataset_path='./../data/'


encoder = Encoder().to(device)
decoder = Decoder(beat=48).to(device)
vae = VAE(encoder, decoder).to(device)

vae.load_state_dict(torch.load("./../models/vae_L1E-02_beta2E+01_beat48_loss2E+01_tanh_gru32_e100_b256_hd64-32_20181008_034323.pt",map_location='cpu'))



data = np.load(dataset_path+'dataset_fill.npz')
X=data['X']
y=data['y']

w=y[:,1,:]

print(w.shape)
# print(w[:300])
index_fills=np.where(w[:,0]==1)

train_x_reduced=X[index_fills[0],:,:]

print(train_x_reduced.shape)



TESTING_RATIO = 0.05
N_DATA = train_x_reduced.shape[0]
N_TRAINING = int(train_x_reduced.shape[0]*TESTING_RATIO)
N_TESTING = N_DATA - N_TRAINING

train_x, test_x = parse_data(train_x_reduced,TESTING_RATIO)
train_dataset = Data.TensorDataset(train_x)
test_dataset = Data.TensorDataset(test_x)

train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)

xs = [np.empty([0])] * 16
ys = [np.empty([0])] * 16
print(len(xs), len(ys))
for batch_i, data in enumerate(train_loader):
    with torch.no_grad():
        # data is a one element array
        print(len(data), "len data")
        data = data[0]
        print(data.shape, "data shape")
        data = Variable(data).type(torch.float32).to(device)
        latent = vae._enc_mu(encoder(data))

        # the distance between first 2 latent vectors
        dist = np.linalg.norm(
            latent[0].cpu().data.numpy() - latent[1].cpu().data.numpy())
        # print('dist: {}'.format(dist))
        latent_numpy = latent.cpu().data.numpy()
        print(latent_numpy.shape, "shape latent")
        # scatter of the latent vectors
        for i in range(16):
            xs[i] = np.concatenate((xs[i], latent.cpu().data.numpy()[:, 2 * i]))
            ys[i] = np.concatenate((ys[i], latent.cpu().data.numpy()[:, 2 * i + 1]))

for i in range(16):
    print(xs[i].shape,ys[i].shape)
    plt.scatter(xs[i], ys[i], s=0.1, alpha=1.0)
    plt.show()