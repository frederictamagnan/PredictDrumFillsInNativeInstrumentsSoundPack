import numpy as np
from RawToOrganized import RawToOrganized
from WideDeepDataset import WideDeepDataset
from WideDeepNet import WideDeepNet
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split

filenpz=np.load('./../data/latentDataset_genre_fill.npz')

BATCH_SIZE=256
N_EPOCHS=400
LR = 0.001
X=filenpz['X']
y=filenpz['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

data_train=RawToOrganized(X_train,y_train)
data_test=RawToOrganized(X_test,y_test)




dataset_train=WideDeepDataset(data_train)
dataset_test=WideDeepDataset(data_test)








train_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=True)



widedeepnet=WideDeepNet(widedata=data_train.wide,deepdata=data_train.deep)
optimizer = optim.Adam(widedeepnet.parameters(), lr=LR)
criterion =  F.binary_cross_entropy




for epoch in range(N_EPOCHS):

    for i, (X_wide, X_deep, target) in enumerate(train_loader):
        X_w = Variable(X_wide).float()
        X_d = Variable(X_deep).float()
        y = Variable(target).float()

        optimizer.zero_grad()
        y_pred = widedeepnet(X_w, X_d)
        print(y_pred.size(),y.size())
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print('Epoch {} of {}, Loss: {}'.format(epoch + 1, N_EPOCHS,loss.data[0]))


correct = 0
total = 0



with torch.no_grad():
    for i, (X_wide, X_deep, target) in enumerate(test_loader):
        X_w = Variable(X_wide).float()
        X_d = Variable(X_deep).float()
        y = Variable(target).float()
        y_pred = widedeepnet(X_w, X_d)
        y_pred_cat = (y_pred > 0.5).squeeze(1).float()

        total += y.size(0)
        correct += (y_pred_cat == y).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))