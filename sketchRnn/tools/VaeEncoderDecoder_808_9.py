
from models.vae_rnn_custom_nine_v2 import *
from tools.VaeDataset import VaeEncodeDataset
from tools.VaeDataset import VaeDecodeDataset
import torch.utils.data as Data
from tools.utils import tensor_to_numpy
import torch
class VaeEncoderDecoder:



    def __init__(self):

        self.batch_size=256
        self.encoder = Encoder(batch_size=256).to(device)
        self.decoder = Decoder(beat=16,batch_size=256).to(device)
        self.vae = VAE(self.encoder, self.decoder,batch_size=256).to(device)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # if self.use_cuda:
        #     print('run on GPU')
        # else:
        #     print('run on CPU')

        self.vae.load_state_dict(torch.load(
            "./../models/vae_L1E-02_beta2E+01_beat8_loss9E+00_tanh_gru64_e45_b8192_hd64-32_20190201_202012.pt",
            map_location=self.device))

        self.count_parameters()

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.vae.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params
        # print(params,"PARAMETERS")




    def pre_process_array(self,array):
        """
        the vae we use only take array whom the length is 256*n
        :param array:
        :return:
        """
        if array.shape[0]% 256 != 0:
            to_complete = 256 - (array.shape[0] % 256)
            blank_lines = np.zeros([to_complete]+list(array.shape)[1:])
            new_array = np.concatenate((array, blank_lines))
        else:
            to_complete=None
            new_array=array

        return new_array,to_complete


    def encode_to_embeddings(self,array):
        """

        :param array: its a batch of reduced drums track with shape n* 96* 9
        :return:
        """
        # print(array.shape,"array shape")
        new_array,to_complete=self.pre_process_array(array)
        # print(new_array.shape,"shape new array")
        # print(to_complete,"to complete")


        train_dataset = VaeEncodeDataset(new_array)
        # print(len(train_dataset),"len dataset")
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=256,
            shuffle=False,
            drop_last=True,
            num_workers=1,
        )

        # warning
        global_tensor = np.zeros((1, 32, 2))

        for batch_i, (data, index) in enumerate(train_loader):
            with torch.no_grad():
                data = Variable(data).type(torch.float32).to(device)
                latent_mu = self.vae._enc_mu(self.encoder(data))
                latent_std = torch.exp(self.vae._enc_log_sigma(self.encoder(data)))
                latent_mu = tensor_to_numpy(latent_mu)
                latent_std = tensor_to_numpy(latent_std)
                latent_tensor = np.stack((latent_mu, latent_std), axis=2)
                global_tensor = np.concatenate((global_tensor, latent_tensor))


        # print(global_tensor.shape,"global tensor shape")
        # print(global_tensor)
        if to_complete is not None:
            global_tensor = global_tensor[1:-to_complete, :, :]
        else:
            global_tensor = global_tensor[1:, :, :]

        # print(global_tensor,"2")
        return global_tensor



    def decode_to_reduced_drums(self,array):

        new_array,to_complete=self.pre_process_array(array)
        test_dataset= VaeDecodeDataset(new_array)
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=1,
        )

        global_tensor = np.zeros((1, 16,9))
        for batch_i, (data) in enumerate(test_loader):
            with torch.no_grad():

                drums_tensor=self.decoder(data)

                global_tensor = np.concatenate((global_tensor, drums_tensor))

        global_tensor = global_tensor[1:-to_complete, :, :]

        return global_tensor


if __name__=='__main__':
    data = np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/vae_dataset.npz')
    data = dict(data)
    vae = data['vae'][:,0,:,:]
    print(vae.shape,"vae shape")

    vaeED=VaeEncoderDecoder()
    tensor=vaeED.decode_to_reduced_drums(vae)
    print(tensor.shape,"WIN")



