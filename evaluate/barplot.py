import matplotlib.pyplot as plt
import numpy as np
names = ['min adj','max adj','mean_adj','count','on 4th beat']

xticks = [i for i in range(len(names))]

data_raw=np.load('metrics_diff.npz')
data=dict(data_raw)


for key in data.keys():
    print(key, data[key][:,5].sum(),data[key][:,6].sum(),data[key][:,7].sum()/data[key][:,5].sum())

for key in data.keys():
    elt=data[key]
    elt=np.mean(elt, axis=0)
    # elt=elt[:-4]*100

    data[key]=elt
title = list(data.keys())
values = list(data.values())

print(values)
# import sys
# sys.exit(0)

# fig, axs = plt.subplots(1, 1,sharey=True)
# plt.subplots_adjust(wspace=0.01)
#
# # for i in range(0,3):
# #     axs[i].bar(xticks, values[i])
# #     axs[i].set_title(title[i])
# #     axs[i].set_xticks(xticks)
# #     axs[i].set_xticklabels(names)
# # fig.suptitle('Fills Statistics')
#
# axs.bar(xticks, values[0])
# axs.set_title(title[0])
# axs.set_xticks(xticks)
# axs.set_xticklabels(names)
# fig.suptitle('Fills Statistics')

# plt.show()



data=dict(data_raw)

genre=['Blues',
'Country',
'Electronic',
'Folk',
'Jazz',
'Latin',
'Metal',
'New-Age',
'Pop',
'Punk',
'Rap',
'Reggae',
'RnB',
    'Rock',
'World',
'Unknown']

dict_count_genre={}
for key in data.keys():
    metrics=data[key]
    for i,elt in enumerate(genre):
        genre_i_metric=metrics[metrics[:,8]==i]
        # print(genre_i_metric.shape,"genre i metric shape")
        genre_i_count=np.mean(genre_i_metric[:,5],axis=0)
        print(genre_i_count)
        dict_count_genre[elt]=genre_i_count

    names= list(dict_count_genre.keys())
    values = list(dict_count_genre.values())


    fig, axs = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
    axs.bar(names, values)
    fig.suptitle(key)

    plt.show()