from DrumReducerExpander import DrumReducerExpander
from models.vae_rnn_custom_nine_v2 import *
from VaeEncoderDecoder_808_9 import VaeEncoderDecoder
class Metadata:




    def __init__(self,batch_multitrack):

        drumReducerExpander=DrumReducerExpander()

        self.batch_multitrack=batch_multitrack
        self.metadata={}


        self.batch_multitrack_reduced_velocity=drumReducerExpander.encode(self.batch_multitrack)
        self.batch_multitrack_reduced_velocity_808=drumReducerExpander.encode_808(self.batch_multitrack_reduced_velocity)
        self.batch_multitrack_reduced=np.zeros(self.batch_multitrack_reduced_velocity.shape)
        self.batch_multitrack_reduced[self.batch_multitrack_reduced_velocity>0]=1
        self.batch_multitrack_reduced_808=drumReducerExpander.encode_808(self.batch_multitrack_reduced)

        self.metadata['vae_embeddings'] = self.vae_embeddings()
        self.metadata['drums_pitches_used'] = self.drum_pitches_used()
        self.metadata['offbeat_notes'] = self.offbeat_notes()
        self.metadata['velocity_metadata'] = self.velocity_metadata()
        self.metadata['diff']=self.diff()

        for key in self.metadata.keys():
            self.metadata[key]=self.metadata[key].reshape((self.metadata[key].shape[0],-1))



    def drum_pitches_used(self):
        return np.max(self.batch_multitrack_reduced,axis=1)

    def offbeat_notes(self):

        sum_axis=np.sum(self.batch_multitrack_reduced,axis=2)
        sum_offbeat=np.sum(sum_axis[:,::3],axis=1)
        return sum_offbeat

    def velocity_metadata(self):

        # min_axis=np.min(self.batch_multitrack_reduced_velocity,axis=1)
        # print(min_axis.shape,"min shape")
        max_axis=np.max(self.batch_multitrack_reduced_velocity_808,axis=1)
        std_axis=np.std(self.batch_multitrack_reduced_velocity_808,axis=1,dtype=np.float64)
        mean_axis=np.mean(self.batch_multitrack_reduced_velocity_808,axis=1,dtype=np.float64)

        velocity_metadata=np.concatenate([max_axis,std_axis,mean_axis],axis=1)
        # print(label_array.shape,"LABEL ARRAY SHAPE")

        return velocity_metadata

    def vae_embeddings(self):
        e=VaeEncoderDecoder()
        emb=e.encode_to_embeddings(self.batch_multitrack_reduced_808)
        print(emb)
        return emb

    def diff(self):

        diff=np.diff(self.batch_multitrack_reduced_808,axis=0)
        print(diff.shape)
        diff=np.concatenate((np.zeros((1,16,9)),diff))
        return diff

    def save_metadata(self,filepath,name):
        np.savez(filepath+name+'_metadata_training.npz',**self.metadata)

if __name__=='__main__':
    lol=np.zeros((257,96,128))
    metadata=Metadata(lol)
    metadata_d=metadata.metadata
    for key in metadata_d.keys():
        print("shape metadata", key, metadata_d[key].shape)












