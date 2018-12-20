import numpy as np
from RawToOrganized import RawToOrganized
from WideDeepDataset import WideDeepDataset
from WideDeepNet import WideDeepNet
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch
from utils import tensor_to_numpy
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import random_split
from sklearn.metrics import confusion_matrix
data_raw=np.load('/home/ftamagna/Documents/_AcademiaSinica/dataset/TotalFills/total.npz')
data_raw=dict(data_raw)
BATCH_SIZE=100
N_EPOCHS=100
LR = 0.001
# X=filenpz['X']
# y=filenpz['y']
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
#
# data_train=RawToOrganized(X_train,y_train)
# data_test=RawToOrganized(X_test,y_test)



dataset=WideDeepDataset(data_raw)

ratio=0.8
train_length= int(len(dataset)* ratio)
test_length=len(dataset)-train_length



train,test=random_split(dataset,[train_length,test_length])






train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=BATCH_SIZE,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=BATCH_SIZE,shuffle=True)



widedeepnet=WideDeepNet(widedata=dataset.X_wide,deepdata=dataset.X_deep)
optimizer = optim.Adam(widedeepnet.parameters(), lr=LR)
criterion =  F.binary_cross_entropy




for epoch in range(N_EPOCHS):

    for i, (X_wide, X_deep, target) in enumerate(train_loader):
        X_w = Variable(X_wide).float()
        X_d = Variable(X_deep).float()
        y = Variable(target).float()

        optimizer.zero_grad()
        y_pred = widedeepnet(X_w, X_d)
        # print(y_pred.size(),y.size())
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if i%1000==0:
            print('Epoch {} of {}, Loss: {}'.format(epoch + 1, N_EPOCHS,loss.data[0]))



model_parameters = filter(lambda p: p.requires_grad, widedeepnet.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("NUMBER OF PARAMETER : "+str(params))




correct = 0
total = 0
y_pred_total=[]
y_true_total=[]


with torch.no_grad():
    for i, (X_wide, X_deep, target) in enumerate(test_loader):
        X_w = Variable(X_wide).float()
        X_d = Variable(X_deep).float()
        y = Variable(target).float()
        y_pred = widedeepnet(X_w, X_d)
        y_pred_cat = (y_pred > 0.65).squeeze(1).float()

        y_pred_total.append(tensor_to_numpy(y_pred_cat).astype(int))
        y_true_total.append(tensor_to_numpy(y).astype(int))

        total += y.size(0)
        correct += (y_pred_cat == y).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

y_pred_total=np.concatenate(y_pred_total)
y_true_total=np.concatenate(y_true_total)

lenpred=len(y_pred_total)
lentrue=len(y_true_total)



conf=confusion_matrix(y_true_total,y_pred_total).ravel()
print(conf/total*100)
tn, fp, fn, tp=conf

print(tp/(fp+tp),"Ratio of True among all fills found")
print(tp/(fn+tp),"Ratio of True among all fills in dataset")


print((tp+fp)/y_true_total.sum(),"Ratio of fills found among all fills")


