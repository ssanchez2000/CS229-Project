import torch
from torch import np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class GenderDataset(Dataset):

    def __init__(self,csv_path,dtype,mode):
        train_data = pd.read_csv(csv_path)
        self.dtype = dtype
        self.mode=mode
        if(mode=="train" or mode=="val"):
            labels=np.ravel(train_data.ix[:,0:1])
            pixels=train_data.ix[:,1:]
            pixels_train, pixels_test, labels_train, labels_test = train_test_split(pixels, labels, random_state=0,train_size=0.9)
            self.N=pixels_train.shape[0]
            self.V=pixels_test.shape[0]
            self.pixels_train=np.array(pixels_train).reshape([self.N,1,28,28])
            self.labels_train=np.array(labels_train).reshape([self.N,1])
            self.labels_test=np.array(labels_test).reshape([self.V,1])
            self.pixels_test=np.array(pixels_test).reshape([self.V,1,28,28])

        if(mode=="test"):
            test_data=pd.read_csv("../input/test.csv")
            self.T=test_data.shape[0]
            self.test=np.array(test_data).reshape([self.T,1,28,28])

    def __getitem__(self,index):
        if(self.mode=="train"):
            label=torch.from_numpy(self.labels_train[index]).type(self.dtype)
            img=torch.from_numpy(self.pixels_train[index]).type(self.dtype)
            return img, label

        if(self.mode=="val"):
            label=torch.from_numpy(self.labels_test[index]).type(self.dtype)
            img=torch.from_numpy(self.pixels_test[index]).type(self.dtype)
            return img,label

        if(self.mode=="test"):
            img=torch.from_numpy(self.test[index]).type(self.dtype)
            return img

    def __len__(self):
        if(self.mode=="train"):
            return self.N
        if(self.mode=="val"):
            return self.V
        if(self.mode=="test"):
            return self.T


dtype = torch.cuda.FloatTensor
save_model_path = "model_state_dict.pkl"
csv_path = '../../data/train_v2.csv'
img_path = '../../data/train-jpg'
img_ext=".jpg"
training_dataset = AmazonDataset(csv_path, img_path, img_ext, dtype)
## loader
train_loader = DataLoader(
    training_dataset,
    batch_size=256,
    shuffle=True, # 1 for CUDA
    #pin_memory=True # CUDA only
)
## simple linear model
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
nn.Linear(1024, 17))

model.type(dtype)
model.train()
loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr=5e-2)
torch.cuda.synchronize()
train(train_loader, model, loss_fn, optimizer, dtype,num_epochs=1, print_every=10)

torch.save(model.state_dict(), save_model_path)
state_dict = torch.load(save_model_path)
model.load_state_dict(state_dict)
