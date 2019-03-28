import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable





class Encoder(torch.nn.Module):

    def __init__(self,num_features=9,gru_hidden_size=64,gru_hidden_size_2=64,seq_len=16,num_directions=2,linear_hidden_size=[64,32],bars_input=2):


        super(Encoder, self).__init__()

        '''
        check the GPU usage
        '''
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # self.device = torch.device("cpu")

        self.bars_input=bars_input

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


        self.gru2= torch.nn.GRU(
            input_size=self.gru_out_dim,
            num_layers=1,
            hidden_size=self.gru_hidden_size_2,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.gru_out_dim_2=self.gru_hidden_size_2

        self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim_2)
        self.linear0 = torch.nn.Linear(
            self.gru_out_dim_2,
            self.linear_hidden_size[0])
        self.bn1 = torch.nn.BatchNorm1d(self.linear_hidden_size[0])

    def forward(self, x):


        hn = torch.zeros(4,x.size()[0], self.gru_hidden_size)

        list_cont=[]
        for i in range(self.bars_input):
            input=x[:,i*16:(i+1)*16,:]
            x2, hn = self.gru(input, hn)
            hn_s=hn.view(x.shape[0], 1,-1)
            list_cont.append(hn_s)


        output_gru1=(torch.cat(list_cont, 1))

        hn2 = torch.zeros(1,x.size()[0], self.gru_hidden_size_2)

        x3,hn2=self.gru2(output_gru1,hn2)
        hn2 = hn2.contiguous().view(
            x.shape[0],
            -1)
        x4 = self.bn0(hn2)
        x4 = torch.tanh(self.linear0(x4))
        x4 = self.bn1(x4)

        return x4


class DecoderFills(torch.nn.Module):
    def __init__(self,linear_hidden_size=[64,32],gru_embedding_hidden_size=16):

        super(DecoderFills, self).__init__()


        '''
        check the GPU usage
        '''
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # self.device = torch.device("cpu")
        self.bars_input=2
        self.bars_output=2
        self.linear_hidden_size=linear_hidden_size
        self.num_features=9
        self.gru_embedding_hidden_size=gru_embedding_hidden_size
        self.gru_embeddings=torch.nn.GRU(
            input_size=self.linear_hidden_size[1],
            num_layers=1,
            hidden_size=gru_embedding_hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        self.gru_embeddings_out_dim=gru_embedding_hidden_size

        self.gru=torch.nn.GRU(
            input_size=self.gru_embeddings_out_dim,
            num_layers=1,
            hidden_size=self.num_features,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )





    def forward(self, z_input,z_output):
        # melody = torch.zeros((x.shape[0], self.seq_len*self.bars_output, self.gru_embeddings_out_dim)).to(self.device)

        z_input=z_input.view(z_input.size()[0],1,z_input.size()[1])
        z_output = z_input.view(z_output.size()[0],1, z_output.size()[1])
        x = torch.cat([z_input, z_output], 1)

        hn1 = torch.zeros(1,z_input.size()[0], self.gru_embedding_hidden_size)
        list_concat=[]
        x1, hn1 = self.gru_embeddings(x, hn1)

        for i in range(self.bars_output):
            hn2 = torch.zeros(1, z_input.size()[0], self.num_features)
            x1_=x1[:, i, :]
            x1_ = x1_.view(x1_.size()[0], 1, x1_.size()[1],)
            input = x1_.repeat(1, 16,1)
            x2,hn=self.gru(input,hn2)
            list_concat.append(x2)

        x = (torch.cat(list_concat,1))

        x=torch.sigmoid(x)
        return x


# class EncoderRegular(torch.nn.Module):
#
#     def __init__(self,num_features=9,gru_hidden_size=64,gru_hidden_size_2=64,seq_len=16,num_directions=2,linear_hidden_size=[64,32]):
#
#
#         super(EncoderRegular, self).__init__()
#
#         '''
#         check the GPU usage
#         '''
#         self.use_cuda = torch.cuda.is_available()
#         self.device = torch.device("cuda" if self.use_cuda else "cpu")
#
#
#
#         self.bars_input=2
#         self.bars_output=2
#         self.num_layers=2
#         self.num_features = num_features
#         self.gru_hidden_size = gru_hidden_size
#         self.gru_hidden_size_2=gru_hidden_size_2
#         self.seq_len = seq_len
#         self.num_directions = num_directions
#         self.linear_hidden_size = linear_hidden_size
#         if self.num_directions == 1:
#             bidirectional = False
#         else:
#             bidirectional = True
#         self.gru = torch.nn.GRU(
#             input_size=self.num_features,
#             num_layers=self.num_layers,
#             hidden_size=self.gru_hidden_size,
#             bias=True,
#             batch_first=True,
#             bidirectional=bidirectional,
#         )
#
#         self.gru_out_dim = self.gru_hidden_size * self.num_directions * self.num_layers
#
#
#         self.gru2=self.gru = torch.nn.GRU(
#             input_size=self.gru_out_dim,
#             num_layers=1,
#             hidden_size=self.gru_hidden_size_2,
#             bias=True,
#             batch_first=True,
#             bidirectional=False,
#         )
#
#         self.gru_out_dim_2=self.gru_hidden_size_2
#
#         self.bn0 = torch.nn.BatchNorm1d(self.gru_out_dim)
#         self.linear0 = torch.nn.Linear(
#             self.gru_out_dim_2,
#             self.linear_hidden_size[0])
#         self.bn1 = torch.nn.BatchNorm1d(self.linear_hidden_size[0])
#
#     def forward(self, x):
#         print(x.size(),"INPUT FORWARD ENC REG")
#         # output_gru1 = torch.zeros((x.shape[0], self.bars_output, self.gru_out_dim)).to(self.device)
#         hn=None
#         list_cont=[]
#         for i in range(self.bars_output):
#             x, hn = self.gru(x[:,i*16:(i+1)*16,:], hn)
#             hn=hn.view(x.shape[0],-1)
#             list_cont.append(hn)
#
#         output_gru1=(torch.cat(list_cont, 1))
#
#         x2,hn2=self.gru(output_gru1,None)
#         hn2 = hn2.contiguous().view(
#             x.shape[0],
#             -1)
#         x = self.bn0(hn2)
#         x = torch.tanh(self.linear0(x))
#         x = self.bn1(x)
#
#         return x


class LongTermNet(nn.Module):

    def __init__(self, encoderFills, decoderFills,encoderRegular):
        super(LongTermNet, self).__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            print('run on GPU')
        else:
            print('run on CPU')
        # self.device = torch.device("cpu")


        self.encoderFills = encoderFills
        self.decoderFills = decoderFills
        self.encoderRegular=encoderRegular
        self.linear_hidden_size=encoderFills.linear_hidden_size
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
        output = self.decoderFills(z_f,z_r)
        return output




    def elbo(self,recon_tracks, tracks, beta=1):
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
        KLD_f = -0.5 * torch.sum(1 + self.log_f - self.mu_f.pow(2) - self.log_f.exp())
        KLD_r = -0.5 * torch.sum(1 + self.log_r - self.mu_r.pow(2) - self.log_r.exp())


        return BCE + KLD_f*beta+KLD_r * beta, BCE, KLD_f,KLD_r








