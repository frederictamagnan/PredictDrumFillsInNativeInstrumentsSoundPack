
import numpy as np
from DrumReducerExpander import DrumReducerExpander



filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumDetection/'
name_raw=['FillsExtractedSupervised.npz','FillsExtractedRuleBased.npz']


for name in name_raw:

    data=np.load(filepath+name)




    #minimum of note
    minn=7
    tr=data['track_array']
    track_array=(tr>0)*1
    print(track_array.shape,"before")
    track_array, indices = np.unique(track_array, axis=0, return_index=True)
    vae = data['vae'][indices]
    genre = data['genre'][indices]
    assert track_array.shape[0]==genre.shape[0] ==vae.shape[0]
    print(track_array.shape[0],"after cleaning unique")
    regular=track_array[:,0,:,:].reshape((track_array.shape[0],16*9))
    fill=track_array[:,1,:,:].reshape((track_array.shape[0],16*9))
    regular2 = track_array[:, 2, :, :].reshape((track_array.shape[0], 16 * 9))
    regular_sum=np.sum(regular,axis=1)
    fill_sum=np.sum(fill,axis=1)
    regular_sum2 = np.sum(regular2, axis=1)
    indices=np.argwhere(np.logical_and(np.logical_and(regular_sum>minn, fill_sum>minn),regular_sum2>minn))
    # break
    vae=vae[indices[:,0],:,:,:]
    track_array=track_array[indices[:,0],:,:,:]
    tr=tr[indices[:,0],:,:,:]
    genre=genre[indices[:,0],:,:]
    assert track_array.shape[0]==genre.shape[0] ==vae.shape[0]
    print(track_array.shape[0],tr.shape[0],"after cleaning min")


    #snare
    snare=track_array[:,1,:,1]
    # print(snare.shape)
    sum_snare=np.sum(snare,axis=1)
    indices = np.argwhere(sum_snare<8)
    # print(indices.shape)
    vae = vae[indices[:, 0], :, :, :]
    track_array = track_array[indices[:, 0], :, :, :]
    tr=tr[indices[:, 0], :, :, :]
    genre = genre[indices[:, 0], :, :]
    assert track_array.shape[0] == genre.shape[0] == vae.shape[0]
    print(track_array.shape[0], "after cleaning snare",tr.shape[0])
    print(track_array.shape)
    np.savez(filepath+name.replace('.npz','_c.npz'),vae=vae,track_array=tr,genre=genre)

