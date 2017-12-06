import torch
from torch import np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class GenderDataset(Dataset):

    def __init__(self,csv_path,file_name,dtype,mode):
        train_data = pd.read_csv(csv_path+file_name)
        self.dtype = dtype
        self.mode=mode
        self.csv_path=csv_path
        if(mode=="train" or mode=="val"):
            labels=train_data.ix[:,5:6]
            img_names=train_data.ix[:,0:1]
            img_names_train, img_names_val, labels_train, labels_val = train_test_split(img_names, labels, random_state=7,train_size=0.85,test_size=0.15)
            self.N=img_names_train.shape[0]
            self.V=img_names_val.shape[0]
            self.img_names_train=np.array(img_names_train).reshape([self.N,1])
            self.labels_train=np.array(labels_train).reshape([self.N,1])
            self.labels_val=np.array(labels_val).reshape([self.V,1])
            self.img_names_val=np.array(img_names_val).reshape([self.V,1])

        if(mode=="test"):
            test_data=pd.read_csv(csv_path+file_name)
            self.T=test_data.shape[0]
            self.img_names_test=np.array(test_data.ix[:,0:1]).reshape([self.T,1])
            self.labels_test=np.array(test_data.ix[:,5:6]).reshape([self.T,1])

    def __getitem__(self,index):
        if(self.mode=="train"):
            label=torch.from_numpy(self.labels_train[index]).type(self.dtype)
            img_name=self.img_names_train[index]
            img=np.array(Image.open(self.csv_path+img_name[0])).T
            img=torch.from_numpy(img).type(self.dtype)
            return img, label

        if(self.mode=="val"):
            label=torch.from_numpy(self.labels_val[index]).type(self.dtype)
            img_name=self.img_names_val[index]
            img=np.array(Image.open(self.csv_path+img_name[0])).T
            img=torch.from_numpy(img).type(self.dtype)
            return img,label

        if(self.mode=="test"):
            label=torch.from_numpy(self.labels_test[index]).type(self.dtype)
            img_name=self.img_names_test[index]
            img=np.array(Image.open(self.csv_path+img_name[0])).T
            img=torch.from_numpy(img).type(self.dtype)
            return img,label

    def __len__(self):
        if(self.mode=="train"):
            return self.N
        if(self.mode=="val"):
            return self.V
        if(self.mode=="test"):
            return self.T

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)


def train(loader_train,val_loader, model, loss_fn, optimizer, dtype,num_epochs=1, print_every=20):
    """
    train `model` on data from `loader_train` for one epoch

    inputs:
    `loader_train` object subclassed from torch.data.DataLoader
    `model` neural net, subclassed from torch.nn.Module
    `loss_fn` loss function see torch.nn for examples
    `optimizer` subclassed from torch.optim.Optimizer
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    acc_history = []
    loss_history = []
    val_acc_history = []
    model.train()
    for i in range(num_epochs):
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())
            y_var=y_var.view(y_var.data.shape[0])
            scores = model(x_var)
            loss = loss_fn(scores, y_var)
            loss_history.append(loss.data[0])

            y_pred = scores.data.max(1)[1].cpu().numpy()
            acc = (y_var.data.cpu().numpy()==y_pred).sum()/float(y_pred.shape[0])
            acc_history.append(acc)

            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f, acc = %.4f' % (t + 1, loss.data[0], acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = validate_epoch(model, val_loader, dtype)
        print('Val accc  %.4f' %  val_acc)
        val_acc_history.append(val_acc)
    return loss_history, acc_history, val_acc_history

def validate_epoch(model, loader, dtype):
    """
    validation for MultiLabelMarginLoss using f2 score

    `model` is a trained subclass of torch.nn.Module
    `loader` is a torch.dataset.DataLoader for validation data
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    n_samples = len(loader.sampler)
    x, y = loader.dataset[0]
    y_array = np.zeros((n_samples))
    y_pred_array = np.zeros((n_samples))
    bs = loader.batch_size
    ## Put the model in test mode
    model.eval()
    for i, (x, y) in enumerate(loader):
        x_var = Variable(x.type(dtype), volatile=True)
        y_var = Variable(y.type(dtype).long())
        y_var=y_var.view(y_var.data.shape[0])
        scores = model(x_var)
        y_pred = scores.data.max(1)[1].cpu().numpy()

        y_array[i*bs:(i+1)*bs] = y_var.data.cpu().numpy()
        y_pred_array[i*bs:(i+1)*bs] = y_pred

    return (y_array==y_pred_array).sum()/float(y_pred_array.shape[0])

dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
train_csv_path = '../../data/train_face/'
train_file_name="gender_fex_trset.csv"
test_csv_path="../../data/test_face/"
test_file_name="gender_fex_valset.csv"
save_model_path="gender_model.pkl"
train_dataset = GenderDataset(train_csv_path, train_file_name, dtype,"train")
## loader
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

val_dataset = GenderDataset(train_csv_path, train_file_name, dtype,"val")
## loader
val_loader = DataLoader(val_dataset,batch_size=64,shuffle=True)

test_dataset = GenderDataset(test_csv_path, test_file_name, dtype,"test")
## loader
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)
print("loaded data")

temp_model=nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(16),
    nn.Conv2d(16, 16, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(16),
    nn.AdaptiveMaxPool2d(128),
    ## 128x128
    nn.Conv2d(16, 32, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.AdaptiveMaxPool2d(64),
    ## 64x64
    nn.Conv2d(32, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.AdaptiveMaxPool2d(32),
    Flatten())

temp_model = temp_model.type(dtype)
temp_model.train()
size=0

for t, (x, y) in enumerate(train_loader):
    x_var = Variable(x.type(dtype))
    size=temp_model(x_var).size()
    if(t==0):
        break

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(16),
    nn.Conv2d(16, 16, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(16),
    nn.AdaptiveMaxPool2d(128),
    ## 128x128
    nn.Conv2d(16, 32, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.AdaptiveMaxPool2d(64),
    ## 64x64
    nn.Conv2d(32, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.AdaptiveMaxPool2d(32),
    Flatten(),
    nn.Linear(size[1], 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096,1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024,2),
    nn.Softmax())
print("defined model")

model.type(dtype)
model.train()
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr=5e-5,weight_decay=1e-1)
print("start training")
loss_history,acc_history,val_acc_history=train(train_loader,val_loader, model, loss_fn, optimizer, dtype,num_epochs=15, print_every=17)

plt.plot(range(len(loss_history)),loss_history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.savefig("gender_loss.png")
plt.gcf().clear()

plt.plot(range(len(acc_history)),acc_history)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.savefig("gender_acc.png")
plt.gcf().clear()

plt.plot(range(len(val_acc_history)),val_acc_history)
plt.xlabel("epochs")
plt.ylabel("val_accuracy")
plt.savefig("gender_acc_val.png")
plt.gcf().clear()

torch.save(model.state_dict(), save_model_path)
state_dict = torch.load(save_model_path)
model.load_state_dict(state_dict)
print("model saved and loaded")
print("start validation")
val_acc=validate_epoch(model, val_loader, dtype)
print(val_acc)
