import torch
from sketchRnn.SketchRnnDataset import SketchRnnDataset
from sketchRnn.SketchRnnNet import SketchRnnNet
from sketchRnn.SketchRnnNet import SketchEncoder
from sketchRnn.SketchRnnNet import SketchDecoder
import numpy as np
from sketchRnn.utils.DrumReducerExpander import DrumReducerExpander
from torch.autograd import Variable
from sketchRnn.utils.utils import tensor_to_numpy
import torch.utils.data

class GeneratorSketchRnn:

    def __init__(self,model_name="sketch_rnn.pt"):

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model_path = "./sketchRnn/model/"
        self.model_name = model_name
        self.linear_hidden_size=[64,32]
        self.gru_hidden_size=64

        self.batch_size=1



    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.params=params


    def load_model(self,batch_size):
        encoder = SketchEncoder(batch_size=batch_size,linear_hidden_size=self.linear_hidden_size, gru_hidden_size=self.gru_hidden_size).to(
            self.device)
        decoder = SketchDecoder(batch_size=batch_size,linear_hidden_size=self.linear_hidden_size).to(self.device)
        self.model = SketchRnnNet(encoder, decoder).to(self.device)
        self.model.eval()

        self.model.load_state_dict(torch.load(self.model_path + self.model_name, map_location="cuda" if self.use_cuda else "cpu"))


    def generate(self,array):




        self.load_model(batch_size=1)
        encoder=DrumReducerExpander()

        X=encoder.encode(array,no_batch=True)
        X=encoder.encode_808(X,no_batch=True)
        X=X.reshape((1,1,16,9))
        X_dataset=SketchRnnDataset(numpy_array=X,inference=True,use_cuda=self.use_cuda)
        X_loader=torch.utils.data.DataLoader(dataset=X_dataset, batch_size=1,
                                                             shuffle=False,drop_last=False)
        with torch.no_grad():
            for i, (x) in enumerate(X_loader):
                x = Variable(x).float()
                y_pred = self.model(x)
                y_pred_cat = (y_pred >0.30)

        y_pred_cat=tensor_to_numpy(y_pred_cat).astype(int)
        # print(new[0].sum())
        new_dec=encoder.decode(y_pred_cat)
        new_dec=encoder.decode_808(new_dec)

        return new_dec.reshape((96,128))



if __name__=='__main__':
    array=np.zeros((96,128))
    g=GeneratorSketchRnn()
    lol=g.generate(array)
