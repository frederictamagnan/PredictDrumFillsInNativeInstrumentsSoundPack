import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable


class SketchEncoder(nn.Module):

    def __init__(self,batch_size=256,num_features=9,gru_hidden_size=64,seq_len=16,num_directions=2,linear_hidden_size=[64,32]):
        super(SketchEncoder, self).__init__()

        self.num_features=num_features
        self.gru_hidden_size=gru_hidden_size
        self.seq_len=seq_len
        self.num_directions=num_directions
        self.linear_hidden_size=linear_hidden_size
        if self.num_directions==1:
            bidirectional=False
        else:
            bidirectional=True
        self.gru = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=1,
            hidden_size=self.gru_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.batch_size = batch_size
        self.gru_out_dim = self.seq_len*self.gru_hidden_size*self.num_directions
        self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)
        self.linear0 = torch.nn.Linear(
            self.gru_out_dim,
            self.linear_hidden_size[0])
        self.bn1 = torch.nn.BatchNorm1d(self.linear_hidden_size[0])


    def forward(self, x):


        x, _ = self.gru(x, None)
        x = x.contiguous().view(
            self.batch_size,
            self.gru_out_dim)
        x = self.bn0(x)
        x = torch.tanh(self.linear0(x))
        x = self.bn1(x)

        return x




class SketchDecoder(nn.Module):

    def __init__(self,batch_size=256,num_features=9,seq_len=16,linear_hidden_size=[64,32]):
        super(SketchDecoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.linear_hidden_size = linear_hidden_size
        self.batch_size=batch_size
        self.gru = torch.nn.GRU(
            input_size=self.num_features+self.linear_hidden_size[1],
            num_layers=1,
            hidden_size=self.linear_hidden_size[1],
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.bn1 = torch.nn.BatchNorm1d(self.seq_len)
        self.bn2 = torch.nn.BatchNorm1d(self.seq_len)
        self.linear1 = torch.nn.Linear(
            self.linear_hidden_size[1],
            self.num_features)

    def forward(self, x,z,hz):
        z_big=z.view(-1, 1,32)
        z_big=z_big.repeat(1, 16,1)

        x = torch.cat([x, z_big], 2)

        # print(x.size()[0],"x size after cat")

        # x = x.contiguous().view(
        #     self.batch_size,
        #     self.seq_len,
        #     self.num_features + self.linear_hidden_size[1])
        # print(x.size(),"contiguous")
        hz=hz.view(-1,x.size()[0],self.linear_hidden_size[1])
        x,hz=self.gru(x,hz)
        # print(x.size(),"X after gru dec")
        x=self.bn1(x)
        # print(x.size(),"X after gru dec")
        x=self.linear1(x)
        # print(x.size(), "X after linear1")
        x = self.bn2(x)
        x = torch.sigmoid(x)

        return x





class SketchRnnNet(nn.Module):

    def __init__(self, encoder, decoder):
        super(SketchRnnNet, self).__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # if use_cuda:
        #     print('run on GPU')
        # else:
        #     print('run on CPU')



        self.encoder = encoder
        self.decoder = decoder
        self.linear_hidden_size=encoder.linear_hidden_size
        self.batch_size = encoder.batch_size
        self._enc_mu = torch.nn.Linear(
            self.linear_hidden_size[0],
            self.linear_hidden_size[1])
        self._enc_log_sigma = torch.nn.Linear(
            self.linear_hidden_size[0],
            self.linear_hidden_size[1])

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(
            np.random.normal(0, 1, size=sigma.size())
        ).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).to(self.device)

    def forward(self, x):
        h_enc = self.encoder(x)
        z = self._sample_latent(h_enc)
        hz=torch.tanh(z)
        output = self.decoder(x,z,hz)

        return output




def elbo(recon_tracks, tracks, mu, sigma, beta=0.5):
    """
    Args:
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
    """
    BCE = F.binary_cross_entropy(
        recon_tracks,
        tracks,
        reduction='sum',
    )
    # KLD = beta * torch.sum(mu * mu + sigma.exp() - sigma - 1)
    KLD = torch.sum(mu * mu + sigma.exp() - sigma - 1)
    return BCE + KLD * beta, BCE, KLD
