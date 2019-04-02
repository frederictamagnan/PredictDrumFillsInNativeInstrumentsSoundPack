import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
import sys

class Visualization:

    def __init__(self,metrics_dir,metrics_filename,label='velocity_metrics'):

        self.metrics_dir=metrics_dir
        self.metrics_filename=metrics_filename

        self.data= np.load(self.metrics_dir + self.metrics_filename)
        self.data = dict(self.data)
        self.genre=self.data['genre']
        self.X= self.data[label]
        self.y = self.data['fills'][:,1].reshape(-1)
        print(self.y.shape)
        print(self.genre.shape)



    def plot_by_genre(self):
        target_ids = range(2)
        target_names = 'regular drum', 'drum fill'

        # plt.figure(figsize=(5, 6))
        colors = 'r', 'g'

        style = ["Pop", "Funk", "Jazz", "Hard", "Metal", "Blues & Country", "Blues Rock", "Ballad", "Indie Rock",
                 "Indie Disco", "Punk Rock"]

        for j, elt in enumerate(style):

            # X_mu=X_mu_raw[indexes]
            print(elt)
            y = self.y[self.genre[:,j]==1]
            X = self.X[self.genre[:, j] == 1]

            X = X[0:500]
            y = y[0:500]

            tsne = TSNE(n_components=2, random_state=0)
            X_trans= tsne.fit_transform(X)
            print(X_trans.shape)
            # X_std_2d = SelectKBest(chi2, k=2).fit_transform(X_mu+0.5, y)
            print(X_trans.shape,"x trans shape")
            for i, c, label in zip(target_ids, colors, target_names):
                plt.scatter(X_trans[y == i, 0], X_trans[y == i, 1], c=c, label=label)
                plt.title(elt+" \n"+"% of fills : " + str(100*y.sum()/len(y)))
            plt.legend()
            plt.show()

    def plot(self):
        target_ids = range(2)
        target_names = 'regular drum', 'drum fill'

        # plt.figure(figsize=(5, 6))
        colors = 'r', 'g'



        X = self.X[0:1500]
        y = self.y[0:1500]

        tsne = TSNE(n_components=2, random_state=0)
        X_trans = tsne.fit_transform(X)
        print(X_trans.shape)
        # X_std_2d = SelectKBest(chi2, k=2).fit_transform(X_mu+0.5, y)
        print(X_trans.shape, "x trans shape")
        for i, c, label in zip(target_ids, colors, target_names):
            plt.scatter(X_trans[y == i, 0], X_trans[y == i, 1], c=c, label=label)
            plt.title("% of fills : " + str(100*y.sum()/len(y)))
        plt.legend()
        plt.show()


if __name__=='__main__':

    metricsdir = '/home/ftamagna/Documents/_AcademiaSinica/dataset/TrainingData/MetricsFiles/'
    metricsfilename = 'total_metrics_training.npz'

    viz=Visualization(metricsdir,metricsfilename)
    # viz.plot()
    viz.plot_by_genre()









