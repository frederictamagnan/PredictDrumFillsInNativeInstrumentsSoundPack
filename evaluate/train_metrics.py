import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()


def training_overview():
    for beta in [1,10,250]:
        for name in ['c','s','d']:
            data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/beta/metrics_'+name+'_'+str(beta)+'.pt.npy')
            # dataset = pd.DataFrame({'Epochs': data[:, 0],'Train Loss': data[:, 1], 'BCE': data[:, 2],'kld': data[:, 3],'Test Loss': data[:, 4]}
            # dataset = pd.DataFrame({'BCE': data[:, 2],'kld': data[:, 3],'Test Loss': data[:, 4]})

            # print(a.shape)
            # a=a.reshape((-1,5))
            # np.savetxt(name+'_'+str(beta)+'loss.csv',a,delimiter=';',newline='\n',fmt="%s")
            overall = pd.DataFrame({'Train Loss': data[:, 1], 'BCE': data[:, 2],'kld': data[:, 3],'Test Loss': data[:, 4]})
            ax = sns.lineplot( data=overall)
            fig=ax.fig


            plt.show()
names=['Clustering','Supervised','Rule Based']
metrics=['Train Loss','BCE','kld','Test Loss']
def influence_beta():
    for i_name,name in enumerate(['c','s','d']):
        list_df = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        for beta in [1,10,250]:
            data=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/trainingMetricsLoss/beta/metrics_'+name+'_'+str(beta)+'.pt.npy')
            for i,elt in enumerate(['Train Loss','BCE','kld','Test Loss']):
                list_df[i]['beta='+str(beta)]=data[:,i+1]

        for i,df in enumerate(list_df):
            ax = sns.lineplot(data=df)
            fig=ax.get_figure()
            fig.suptitle(names[i_name]+': '+metrics[i]+" with respect to beta")
            plt.show()


if __name__=='__main__':
    influence_beta()