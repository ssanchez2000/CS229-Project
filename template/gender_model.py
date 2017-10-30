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

class GenderDataset(Dataset):

    def __init__(self,csv_path,file_name,dtype,mode):
        train_data = pd.read_csv(csv_path+file_name)
        self.dtype = dtype
        self.mode=mode
        self.csv_path=csv_path
        if(mode=="train" or mode=="val"):
            labels=train_data.ix[:,5:6]
            img_names=train_data.ix[:,0:1]
            img_names_train, img_names_val, labels_train, labels_val = train_test_split(img_names, labels, random_state=0,train_size=0.7,test_size=0.3)
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
            img=np.array(Image.open(csv_path+img_name))
            img=torch.from_numpy(img).type(self.dtype)
            return img, label

        if(self.mode=="val"):
            label=torch.from_numpy(self.labels_val[index]).type(self.dtype)
            img_name=self.img_names_val[index]
            img=np.array(Image.open(csv_path+img_name))
            img=torch.from_numpy(img).type(self.dtype)
            return img,label

        if(self.mode=="test"):
            label=torch.from_numpy(self.labels_test[index]).type(self.dtype)
            img_name=self.img_names_test[index]
            img=np.array(Image.open(csv_path+img_name))
            img=torch.from_numpy(img).type(self.dtype)
            return img,label

    def __len__(self):
        if(self.mode=="train"):
            return self.N
        if(self.mode=="val"):
            return self.V
        if(self.mode=="test"):
            return self.T


dtype = torch.FloatTensor
save_model_path = "model_state_dict.pkl"
csv_path = '../data/smiles_trset/'
file_name="gender_fex_trset.csv"
training_dataset = GenderDataset(csv_path, file_name, dtype,"train")
## loader
train_loader = DataLoader(
    training_dataset,
    batch_size=256,
    shuffle=True, # 1 for CUDA
    #pin_memory=True # CUDA only
)
## simple linear model
"""
temp_model=nn.Sequential(
    nn.Conv2d(4, 16, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(16),
    nn.AdaptiveMaxPool2d(128),
    nn.Conv2d(16, 32, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.AdaptiveMaxPool2d(64),
    Flatten())

temp_model = temp_model.type(dtype)
temp_model.train()
size=0
print(type(train_loader))
for t, (x, y) in enumerate(train_loader):
	x_var = Variable(x.type(dtype)).cuda()
	size=temp_model(x_var).size()
	if(t==0):
		break

model = nn.Sequential(
nn.Conv2d(4, 16, kernel_size=3, stride=1),
nn.ReLU(inplace=True),
nn.BatchNorm2d(16),
nn.AdaptiveMaxPool2d(128),
nn.Conv2d(16, 32, kernel_size=3, stride=1),
nn.ReLU(inplace=True),
nn.BatchNorm2d(32),
nn.AdaptiveMaxPool2d(64),
Flatten(),
nn.Linear(size[1], 1024),
nn.ReLU(inplace=True),
nn.Linear(1024, 1))

model.type(dtype)
model.train()
loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr=5e-2)
torch.cuda.synchronize()
train(train_loader, model, loss_fn, optimizer, dtype,num_epochs=1, print_every=10)

torch.save(model.state_dict(), save_model_path)
state_dict = torch.load(save_model_path)
model.load_state_dict(state_dict)
"""
