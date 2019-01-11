from Metrics import Metrics
import numpy as np
import json



class MetricsTraining(Metrics):



    def __init__(self,batch_multitrack,list_filepath):

        Metrics.__init__(self,batch_multitrack=batch_multitrack)
        self.list_filepath=list_filepath
        self.metrics['list_filepath']=list_filepath
        self.list_genre=["Pop","Funk","Jazz","Hard","Metal","Blues & Country","Blues Rock","Ballad","Indie Rock","Indie Disco","Punk Rock"]
        self.metadata_metrics()


    def metadata_metrics(self):
        self.metrics['fills'] = np.zeros((1, 2, 1))
        self.metrics['genre'] = np.zeros((1, 12, 1))
        self.metrics['bpm'] = np.zeros((1, 1))
        self.metrics['dataset'] = np.zeros((1, 2, 1))

        for filepath in self.list_filepath:
            print("lol")
            # Process the fills label
            if "Fill" in filepath or "OddGrooves" in filepath:
                label_fills = np.array([[0], [1]])
            else:
                label_fills = np.array([[1], [0]])

            # process the genre label
            if "OddGrooves" in filepath:
                label_genre = np.zeros((12, 1))
                label_genre[11] = 1
            else:
                label_genre = self.return_label_genre(filepath)

            # process the BPM and type

            if "OddGrooves" in filepath:
                index_bpm = filepath.index("BPM")
                index_middle = filepath.index("-")
                bpm = int(filepath[index_middle + 3:index_bpm - 1])
                label_bpm = np.full(shape=(1, 1), fill_value=bpm)
                label_dataset = np.array([[0], [1]])

            else:
                list_ = ['Groove', 'Fill']
                offset = [8, 6]
                index_ = int(label_fills[1, 0])
                print(index_)
                string_ = list_[index_]
                offset_ = offset[index_]
                index_bpm = filepath.index("BPM")
                print(string_)
                index_middle = filepath.index(string_)
                bpm = int(filepath[index_middle + offset_:index_bpm])
                label_bpm = np.full(shape=(1, 1), fill_value=bpm)

                label_dataset = np.array([[1], [0]])

            label_genre = label_genre.reshape(1, label_genre.shape[0], label_genre.shape[1])
            label_fills = label_fills.reshape(1, label_fills.shape[0], label_fills.shape[1])
            label_dataset = label_dataset.reshape(1, label_dataset.shape[0], label_dataset.shape[1])

            self.metrics['fills'] = np.concatenate((self.metrics['fills'] , label_fills))
            self.metrics['genre'] = np.concatenate((self.metrics['genre'] , label_genre))
            self.metrics['bpm'] = np.concatenate((self.metrics['bpm'], label_bpm))
            self.metrics['dataset'] = np.concatenate((self.metrics['dataset'] , label_dataset))

        list_special_label=['fills',"genre","bpm","dataset"]
        for key in list_special_label:
            self.metrics[key]=np.delete(self.metrics[key], 0, 0)
            if key!='fills':
                self.metrics[key] = self.metrics[key].reshape((self.metrics[key].shape[0], -1))




    def return_label_genre(self,subdir):

        label_genre=np.zeros((12,1))

        for i,elt in enumerate(self.list_genre):

            if elt in subdir:
                label_genre[i,0]=1
                print("RETURN LABEL")
                return label_genre

        raise "Error finding genre"

    def save_metrics(self,filepath,name):
        with open(filepath +name +'_list_filepath' + '.json', 'w') as outfile:
            json.dump(self.metrics['list_filepath'], outfile)

        self.metrics.pop('list_filepath', None)
        super().save_metrics(filepath,name)
