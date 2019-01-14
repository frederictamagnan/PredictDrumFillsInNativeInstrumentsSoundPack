from DrumReducerExpander import DrumReducerExpander
from sklearn.preprocessing import binarize
from models.vae_rnn import *
from DrumsDataset import DrumsDataset
import torch.utils.data as Data
from utils import tensor_to_numpy

class Metrics:




    def __init__(self,batch_multitrack):

        drumReducerExpander=DrumReducerExpander()

        self.batch_multitrack=batch_multitrack
        self.metrics={}
        remove_blank_lines=False

        if len(self.batch_multitrack)%256!=0:

            to_complete=256-(len(self.batch_multitrack)%256)
            self.blank_lines=np.zeros((to_complete,96,128))
            self.batch_multitrack=np.concatenate((self.batch_multitrack,self.blank_lines))
            remove_blank_lines=True
            print(self.batch_multitrack.shape)

        self.batch_multitrack_reduced_velocity=drumReducerExpander.encode(self.batch_multitrack)
        print(self.batch_multitrack_reduced_velocity.shape)
        self.batch_multitrack_reduced=np.zeros(self.batch_multitrack_reduced_velocity.shape)
        self.batch_multitrack_reduced[self.batch_multitrack_reduced_velocity>0]=1

        self.metrics['vae_embeddings'] = self.vae_embeddings()
        self.metrics['drums_pitches_used'] = self.drum_pitches_used()
        self.metrics['offbeat_notes'] = self.offbeat_notes()
        self.metrics['velocity_metrics'] = self.velocity_metrics()

        for key in self.metrics.keys():
            self.metrics[key]=self.metrics[key].reshape((self.metrics[key].shape[0],-1))


        if remove_blank_lines:
            for key in self.metrics.keys():
                self.metrics[key]=self.metrics[key][:-self.blank_lines.shape[0]]




    def drum_pitches_used(self):
        return np.max(self.batch_multitrack_reduced,axis=1)

    def offbeat_notes(self):

        sum_axis=np.sum(self.batch_multitrack_reduced,axis=2)
        sum_offbeat=np.sum(sum_axis[:,::3],axis=1)
        return sum_offbeat

    def velocity_metrics(self):

        min_axis=np.min(self.batch_multitrack_reduced_velocity,axis=1)
        print(min_axis.shape,"min shape")
        max_axis=np.max(self.batch_multitrack_reduced_velocity,axis=1)
        std_axis=np.std(self.batch_multitrack_reduced_velocity,axis=1)
        mean_axis=np.std(self.batch_multitrack_reduced_velocity,axis=1)

        velocity_metrics=np.concatenate([min_axis,max_axis,std_axis,mean_axis],axis=1)
        # print(label_array.shape,"LABEL ARRAY SHAPE")

        return velocity_metrics

    def vae_embeddings(self):

        BATCH_SIZE = 256
        encoder = Encoder().to(device)
        decoder = Decoder(beat=48).to(device)
        vae = VAE(encoder, decoder).to(device)

        vae.load_state_dict(torch.load(
            "./../models/vae_L1E-02_beta2E+01_beat48_loss2E+01_tanh_gru32_e100_b256_hd64-32_20181008_034323.pt",
            map_location='cpu'))

        train_dataset = DrumsDataset(self.batch_multitrack_reduced)
        print(len(train_dataset))

        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=1,
        )

        # warning
        global_tensor = np.zeros((1, 32, 2))
        global_indexes = np.zeros((1))

        for batch_i, (data, index) in enumerate(train_loader):
            with torch.no_grad():
                data = Variable(data).type(torch.float32).to(device)
                latent_mu = vae._enc_mu(encoder(data))
                latent_std = torch.exp(vae._enc_log_sigma(encoder(data)))
                print(latent_std.shape, latent_mu.shape, "SHAPE LATENT")
                # the distance between first 2 latent vectors

                indexes = tensor_to_numpy(index)

                latent_mu = tensor_to_numpy(latent_mu)
                latent_std = tensor_to_numpy(latent_std)

                latent_tensor = np.stack((latent_mu, latent_std), axis=2)

                global_indexes = np.concatenate((global_indexes, indexes))

                global_tensor = np.concatenate((global_tensor, latent_tensor))

                print(batch_i)

        print(global_indexes.shape, "labels shape")
        print(global_tensor.shape, "global tensor shape")

        global_indexes = global_indexes[1:]
        global_tensor = global_tensor[1:, :, 0]

        return global_tensor


    def save_metrics(self,filepath,name):
        np.savez(filepath+name+'_metrics_training.npz',**self.metrics)

if __name__=='__main__':
    lol=np.zeros((8000,96,128))
    metrics=Metrics(lol)
    metrics_d=metrics.metrics
    for key in metrics_d.keys():
        print("shape metrics", key, metrics_d[key].shape)












