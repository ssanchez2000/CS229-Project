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
class AllDataset(Dataset):

    def __init__(self,csv_path,file_name,dtype,mode):
        train_data = pd.read_csv(csv_path+file_name,header=0, skiprows=[0])
        self.dtype = dtype
        self.mode=mode
        self.csv_path=csv_path
        if(mode=="train" or mode=="val"):
            labels=train_data.ix[:,5:7]
            img_names=train_data.ix[:,0:1]
            img_names_train, img_names_val, labels_train, labels_val = train_test_split(img_names, labels, random_state=0,train_size=0.7,test_size=0.3)
            self.N=img_names_train.shape[0]
            self.V=img_names_val.shape[0]
            self.img_names_train=np.array(img_names_train).reshape([self.N,1])
            self.labels_train=np.array(labels_train).reshape([self.N,2])
            self.labels_val=np.array(labels_val).reshape([self.V,2])
            self.img_names_val=np.array(img_names_val).reshape([self.V,1])

        if(mode=="test"):
            test_data=pd.read_csv(csv_path+file_name,header=0, skiprows=[0])
            self.T=test_data.shape[0]
            self.img_names_test=np.array(test_data.ix[:,0:1]).reshape([self.T,1])
            self.labels_test=np.array(test_data.ix[:,5:7]).reshape([self.T,2])

    def __getitem__(self,index):
        if(self.mode=="train"):
            label=torch.from_numpy(self.labels_train[index]).type(self.dtype)
            img_name=self.img_names_train[index]
            img=Image.open(self.csv_path+img_name[0])
            img=np.array(img).T
            img=torch.from_numpy(img).type(self.dtype)
            return img, label[0],label[1] # gender, smile

        if(self.mode=="val"):
            label=torch.from_numpy(self.labels_val[index]).type(self.dtype)
            img_name=self.img_names_val[index]
            img=Image.open(self.csv_path+img_name[0])
            img=np.array(img).T
            img=torch.from_numpy(img).type(self.dtype)
            return img,label[0],label[1] # gender, smile

        if(self.mode=="test"):
            label=torch.from_numpy(self.labels_test[index]).type(self.dtype)
            img_name=self.img_names_test[index]
            img=Image.open(self.csv_path+img_name[0])
            img=np.array(img).T
            img=torch.from_numpy(img).type(self.dtype)
            return img,label[0],label[1] # gender, smile

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

def all_train(loader_train, all_model,gender_model,smile_model, loss_fn, all_optimizer,gender_optimizer,smile_optimizer, dtype,num_epochs=1, print_every=20):
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
    acc_gender_history = []
    loss_gender_history = []
    acc_smile_history = []
    loss_smile_history = []
    acc_all_history = []
    loss_all_history = []

    all_model.train()

    for i in range(num_epochs):
        for t, (x,y,z) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))

            y_var = Variable(y.type(dtype).long())
            y_var=y_var.view(y_var.data.shape[0])

            z_var = Variable(z.type(dtype).long())
            z_var=z_var.view(z_var.data.shape[0])

            noise_x = all_model(x_var)
            scores_gender=gender_model(noise_x)
            scores_smile=smile_model(noise_x)

            loss_gender = loss_fn(scores_gender, y_var)
            loss_gender_history.append(loss_gender.data[0])

            loss_smile = loss_fn(scores_smile, z_var)
            loss_smile_history.append(loss_smile.data[0])

            loss_all=loss_smile/loss_gender
            loss_all_history.append(loss_all.data[0])

            y_pred = scores_gender.data.max(1)[1].cpu().numpy()
            z_pred = scores_smile.data.max(1)[1].cpu().numpy()

            acc_gender = (y_var.data.cpu().numpy()==y_pred).sum()/float(y_pred.shape[0])
            acc_gender_history.append(acc_gender)

            acc_smile = (z_var.data.cpu().numpy()==z_pred).sum()/float(z_pred.shape[0])
            acc_smile_history.append(acc_smile)

            y_bool=(y_var.data.cpu().numpy()!=y_pred)
            z_bool=(z_var.data.cpu().numpy()==z_pred)
            acc_all=np.multiply(y_bool,z_bool).sum()/float(y_bool.shape[0])
            acc_all_history.append(acc_all)
            if (t + 1) % print_every == 0:
                print('t = %d, loss_all = %.4f,loss_gender = %.4f,loss_smile = %.4f, acc_all = %.4f' % (t + 1, loss_all.data[0],loss_gender.data[0],loss_smile.data[0], acc_all))

            gender_optimizer.zero_grad()
            loss_gender.backward(retain_graph=True)
            gender_optimizer.step()

            smile_optimizer.zero_grad()
            loss_smile.backward(retain_graph=True)
            smile_optimizer.step()

            all_optimizer.zero_grad()
            loss_all.backward()
            all_optimizer.step()

    return loss_all_history, loss_gender_history,loss_smile_history, acc_all_history, acc_gender_history,acc_smile_history



def validate(all_model,gender_model,smile_model, loader, dtype):
    """
    validation for MultiLabelMarginLoss using f2 score

    `model` is a trained subclass of torch.nn.Module
    `loader` is a torch.dataset.DataLoader for validation data
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    n_samples = len(loader.sampler)
    #x, y,z = loader.dataset[0]
    y_array = np.zeros((n_samples))
    y_pred_array = np.zeros((n_samples))
    z_array = np.zeros((n_samples))
    z_pred_array = np.zeros((n_samples))

    bs = loader.batch_size
    ## Put the model in test mode
    all_model.eval()
    gender_model.eval()
    smile_model.eval()
    for i, (x, y, z) in enumerate(loader):
        x_var = Variable(x.type(dtype),volatile=True)

        y_var = Variable(y.type(dtype).long())
        y_var=y_var.view(y_var.data.shape[0])

        z_var = Variable(z.type(dtype).long())
        z_var=z_var.view(z_var.data.shape[0])

        noise_x = all_model(x_var)
        scores_gender=gender_model(noise_x)
        scores_smile=smile_model(noise_x)

        y_pred = scores_gender.data.max(1)[1].cpu().numpy()
        z_pred = scores_smile.data.max(1)[1].cpu().numpy()

        y_array[i*bs:(i+1)*bs] = y_var.data.cpu().numpy()
        y_pred_array[i*bs:(i+1)*bs] = y_pred

        z_array[i*bs:(i+1)*bs] = z_var.data.cpu().numpy()
        z_pred_array[i*bs:(i+1)*bs] = z_pred

    acc_gender = (y_array==y_pred_array).sum()/float(y_pred_array.shape[0])
    acc_smile = (z_array==z_pred_array).sum()/float(z_pred_array.shape[0])

    y_bool=(y_array==y_pred_array)
    z_bool=(z_array==z_pred_array)
    acc_all=np.multiply(y_bool,z_bool).sum()/float(y_bool.shape[0])

    return acc_all,acc_gender,acc_smile

dtype = torch.cuda.FloatTensor
train_csv_path = '../../data/train_face/'
train_file_name="gender_fex_trset.csv"
test_csv_path="../../data/test_face/"
test_file_name="gender_fex_valset.csv"
save_model_path="all_model.pkl"
save_gender_model_path="all_gender_model.pkl"
save_smile_model_path="all_smile_model.pkl"


train_dataset = AllDataset(train_csv_path, train_file_name, dtype,"train")
## loader
train_loader = DataLoader(train_dataset,batch_size=256,shuffle=True)

val_dataset = AllDataset(train_csv_path, train_file_name, dtype,"val")
## loader
val_loader = DataLoader(val_dataset,batch_size=256,shuffle=True)

test_dataset = AllDataset(test_csv_path, test_file_name, dtype,"test")
## loader
test_loader = DataLoader(test_dataset,batch_size=256,shuffle=True)
print("loaded data")


all_temp_model=nn.Sequential(
    Flatten())

all_temp_model = all_temp_model.type(dtype)
all_temp_model.train()
size=0

for t, (x, y, z) in enumerate(train_loader):
    x_var = Variable(x.type(dtype))
    size=all_temp_model(x_var).size()
    if(t==0):
        break

all_model = nn.Sequential(
Flatten(),
nn.Linear(size[1], 256),
nn.ReLU(inplace=True),
nn.Linear(256, size[1]))
all_model.type(dtype)
all_model.train()
print("defined all model")

gender_model= nn.Sequential(
nn.Linear(size[1], 10),
nn.ReLU(inplace=True),
nn.Linear(10, 2))
gender_model.type(dtype)
gender_model.train()
print("defined gender model")

smile_model= nn.Sequential(
nn.Linear(size[1], 10),
nn.ReLU(inplace=True),
nn.Linear(10, 2))
smile_model.type(dtype)
smile_model.train()
print("defined smile model")

loss_fn = nn.CrossEntropyLoss().type(dtype)
all_optimizer = optim.Adam(all_model.parameters(), lr=5e-2)
gender_optimizer = optim.Adam(gender_model.parameters(), lr=5e-2)
smile_optimizer = optim.Adam(smile_model.parameters(), lr=5e-2)
print("start training")
loss_all_history, loss_gender_history,loss_smile_history, acc_all_history, acc_gender_history,acc_smile_history=all_train(train_loader, all_model,gender_model,smile_model, loss_fn, all_optimizer,gender_optimizer,smile_optimizer, dtype,num_epochs=1, print_every=5)

plt.plot(range(len(loss_smile_history)),loss_smile_history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.savefig("smile_loss_all.png")
plt.gcf().clear()

plt.plot(range(len(acc_smile_history)),acc_smile_history)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.savefig("smile_acc_all.png")
plt.gcf().clear()

plt.plot(range(len(loss_gender_history)),loss_gender_history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.savefig("gender_loss_all.png")
plt.gcf().clear()

plt.plot(range(len(acc_gender_history)),acc_gender_history)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.savefig("gender_acc_all.png")
plt.gcf().clear()

plt.plot(range(len(loss_all_history)),loss_all_history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.savefig("all_loss.png")
plt.gcf().clear()

plt.plot(range(len(acc_all_history)),acc_all_history)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.savefig("all_acc.png")
plt.gcf().clear()

torch.save(all_model.state_dict(), save_model_path)
torch.save(gender_model.state_dict(), save_gender_model_path)
torch.save(smile_model.state_dict(), save_smile_model_path)

state_all_dict = torch.load(save_model_path)
all_model.load_state_dict(state_all_dict)

state_gender_dict = torch.load(save_gender_model_path)
gender_model.load_state_dict(state_gender_dict)

state_smile_dict = torch.load(save_smile_model_path)
smile_model.load_state_dict(state_smile_dict)

print("model saved and loaded")
print("start validation")

acc_all,acc_gender,acc_smile=validate(all_model,gender_model,smile_model, val_loader, dtype)
print(acc_all,acc_gender,acc_smile)
