import torch
from torch import np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)

class Unflatten(nn.Module):
    def forward(self, x):
        C=3
        H=256
        W=256
        N,M = x.size() # read in N, C* H* W
        return x.view(N,C,H,W)

def gender_model(size):
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
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 2),
        nn.Softmax())
    return model

def temp_gender():
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
        Flatten())
    return model

def smile_model(size):
    return gender_model(size)

def temp_smile():
    return temp_gender()

def temp_encoder():
    model = nn.Sequential(
        Flatten())
    return model

def encoder_model(size):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(size[1], 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, size[1]),
        Unflatten())
    return model
