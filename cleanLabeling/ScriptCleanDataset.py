import numpy as np
filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/'
filename='FillsExtractedFour.npz'
data=np.load(filepath+filename)
data=dict(data)

minn=7
tr=data['track_array']
tr,indices=np.unique(tr,axis=0,return_index=True)
vae=data['vae'][indices]
genre=data['genre'][indices]

tr=tr>0
regular=tr[:,0,:,:].reshape((tr.shape[0],16*9))
fill=tr[:,3,:,:].reshape((tr.shape[0],16*9))
regular_sum=np.sum(regular,axis=1)
fill_sum=np.sum(fill,axis=1)
indices=np.argwhere(np.logical_and(regular_sum>minn, fill_sum>minn))


vae=vae[indices[:,0],:,:,:]
track_array=tr[indices[:,0],:,:,:]
print(vae.shape,"lol")
genre=genre[indices[:,0],:,:]

print(vae.shape,track_array.shape,genre.shape)

np.savez(filepath+filename.replace('.npz','_cc.npz'),vae=vae,track_array=track_array,genre=genre)
