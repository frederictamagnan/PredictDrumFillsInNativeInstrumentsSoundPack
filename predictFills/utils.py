import torch.utils.data as Data
from DrumsDataset import DrumsDataset
from models.vae_rnn import *



def product(tuple1):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tuple1:
        prod = prod * x
    return prod



def x_encoding(X):


    BATCH_SIZE=256
    encoder = Encoder().to(device)
    decoder = Decoder(beat=48).to(device)
    vae = VAE(encoder, decoder).to(device)

    vae.load_state_dict(torch.load("./../models/vae_L1E-02_beta2E+01_beat48_loss2E+01_tanh_gru32_e100_b256_hd64-32_20181008_034323.pt",map_location='cpu'))

    train_dataset = DrumsDataset(X)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )


    #warning
    global_tensor=np.zeros((1,32,2))
    global_indexes=np.zeros((1))

    for batch_i, (data,index) in enumerate(train_loader):
        with torch.no_grad():

            data = Variable(data).type(torch.float32).to(device)
            latent_mu = vae._enc_mu(encoder(data))
            latent_std=torch.exp(vae._enc_log_sigma(encoder(data)))
            print(latent_std.shape,latent_mu.shape,"SHAPE LATENT")
            # the distance between first 2 latent vectors

            indexes=tensor_to_numpy(index)

            latent_mu=tensor_to_numpy(latent_mu)
            latent_std=tensor_to_numpy(latent_std)


            latent_tensor=np.stack((latent_mu,latent_std),axis=2)


            global_indexes=np.concatenate((global_indexes,indexes))

            global_tensor=np.concatenate((global_tensor,latent_tensor))





            print(batch_i)

    print(global_indexes.shape,"labels shape")
    print(global_tensor.shape,"global tensor shape")


    global_indexes=global_indexes[1:]
    global_tensor=global_tensor[1:,:,:]

    return global_tensor.reshape((global_tensor.shape[0],-1)),global_indexes.reshape(global_indexes.shape[0]).astype(int)




def tensor_to_numpy(array):

    return array.cpu().data.numpy()



