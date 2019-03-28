import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable





class EncoderFills(torch.nn.Module):

    def __init__(self,num_features=9,gru_hidden_size=64,gru_hidden_size_2=64,seq_len=16,num_directions=2,linear_hidden_size=[64,32]):


        super(EncoderFills, self).__init__()

        '''
        check the GPU usage
        '''
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")



        self.bars_input=2
        self.bars_output=2
        self.num_layers=2
        self.num_features = num_features
        self.gru_hidden_size = gru_hidden_size
        self.gru_hidden_size_2=gru_hidden_size_2
        self.seq_len = seq_len
        self.num_directions = num_directions
        self.linear_hidden_size = linear_hidden_size
        if self.num_directions == 1:
            bidirectional = False
        else:
            bidirectional = True
        self.gru = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=self.num_layers,
            hidden_size=self.gru_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.gru_out_dim = self.gru_hidden_size * self.num_directions * self.num_layers


        self.gru2=self.gru = torch.nn.GRU(
            input_size=self.gru_out_dim,
            num_layers=1,
            hidden_size=self.gru_hidden_size_2,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.gru_out_dim_2=self.gru_hidden_size_2

        self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)
        self.linear0 = torch.nn.Linear(
            self.gru_out_dim_2,
            self.linear_hidden_size[0])
        self.bn1 = torch.nn.BatchNorm1d(self.linear_hidden_size[0])

    def forward(self, x):

        output_gru1 = torch.zeros((x.shape[0], self.bars_output, self.gru_out_dim)).to(self.device)
        hn=None
        for i in range(self.bars_output):
            x, hn = self.gru(x, hn)
            output_gru1[:,i:(i+1),:]=hn

        x2,hn2=self.gru(output_gru1,None)
        hn2 = hn2.contiguous().view(
            self.batch_size,
            self.gru_out_dim_2)
        x = self.bn0(hn2)
        x = torch.tanh(self.linear0(x))
        x = self.bn1(x)

        return x


class DecoderFills(torch.nn.Module):
    def __init__(self,linear_hidden_size=[64,32],gru_embedding_hidden_size=16):
        super(DecoderFills, self).__init__()
        '''
                check the GPU usage
                '''
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.bars_input=2
        self.bars_output=2
        self.linear_hidden_size=linear_hidden_size
        self.num_features=9

        self.gru_embeddings(torch.nn.GRU(
            input_size=self.linear_hidden_size[2]*2,
            num_layers=1,
            hidden_size=gru_embedding_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=False,
        ))

        self.gru_embeddings_out_dim=gru_embedding_hidden_size

        self.gru(torch.nn.GRU(
            input_size=self.gru_embeddings_out_dim,
            num_layers=1,
            hidden_size=self.num_features,
            bias=True,
            batch_first=True,
            bidirectional=False,
        ))





    def forward(self, z_input,z_output):
        # melody = torch.zeros((x.shape[0], self.seq_len*self.bars_output, self.gru_embeddings_out_dim)).to(self.device)
        x = torch.cat([z_input, z_output], 1)

        hn1=None
        list_cont=[]
        for i in range(self.bars_output):
            x1,hn1=self.gru_embeddings(x,hn1)
            x2,hn=self.gru(x1.repeat(1, 16, 1),None)

            list_cont.append(x2)

        x = (torch.cat(list_cont,1))

        x=torch.simoid(x)
        return x


class EncoderRegular(torch.nn.Module):

    def __init__(self,num_features=9,gru_hidden_size=64,gru_hidden_size_2=64,seq_len=16,num_directions=2,linear_hidden_size=[64,32]):


        super(EncoderRegular, self).__init__()

        '''
        check the GPU usage
        '''
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")



        self.bars_input=2
        self.bars_output=2
        self.num_layers=2
        self.num_features = num_features
        self.gru_hidden_size = gru_hidden_size
        self.gru_hidden_size_2=gru_hidden_size_2
        self.seq_len = seq_len
        self.num_directions = num_directions
        self.linear_hidden_size = linear_hidden_size
        if self.num_directions == 1:
            bidirectional = False
        else:
            bidirectional = True
        self.gru = torch.nn.GRU(
            input_size=self.num_features,
            num_layers=self.num_layers,
            hidden_size=self.gru_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.gru_out_dim = self.gru_hidden_size * self.num_directions * self.num_layers


        self.gru2=self.gru = torch.nn.GRU(
            input_size=self.gru_out_dim,
            num_layers=1,
            hidden_size=self.gru_hidden_size_2,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.gru_out_dim_2=self.gru_hidden_size_2

        self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)
        self.linear0 = torch.nn.Linear(
            self.gru_out_dim_2,
            self.linear_hidden_size[0])
        self.bn1 = torch.nn.BatchNorm1d(self.linear_hidden_size[0])

    def forward(self, x):

        output_gru1 = torch.zeros((x.shape[0], self.bars_output, self.gru_out_dim)).to(self.device)
        hn=None
        for i in range(self.bars_output):
            x, hn = self.gru(x, hn)
            output_gru1[:,i:(i+1),:]=hn

        x2,hn2=self.gru(output_gru1,None)
        hn2 = hn2.contiguous().view(
            self.batch_size,
            self.gru_out_dim_2)
        x = self.bn0(hn2)
        x = torch.tanh(self.linear0(x))
        x = self.bn1(x)

        return x


class LongTermNet(nn.Module):

    def __init__(self, encoderFills, decoderFills,encoderRegular):
        super(LongTermNet, self).__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')



        self.encoderFills = encoderFills
        self.decoderFills = decoderFills
        self.encoderRegular=encoderRegular
        self.linear_hidden_size=encoderFills.linear_hidden_size
        self.batch_size = encoderFills.batch_size
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
        sigma = torch.exp(log_sigma/2)
        std_z = torch.from_numpy(
            np.random.normal(0, 1, size=sigma.size())
        ).float()


        return mu + sigma * Variable(std_z, requires_grad=False).to(self.device),mu,log_sigma

    def forward(self, regular_pattern,drum_fills):
        h_fills=self.encoderFills(drum_fills)
        h_regular=self.encoderRegular(regular_pattern)
        z_f,mu_f,log_f = self._sample_latent(h_fills)
        z_r,mu_r,log_r = self._sample_latent(h_regular)

        self.mu_f=mu_f
        self.mu_r=mu_r
        self.log_f=log_f
        self.log_r=log_r
        output = self.decoder(z_f,z_r)

        return output




def elbo(recon_tracks, tracks, mu, logvar, beta=1):
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
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD * beta, BCE, KLD








