import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms

import numpy as np
from Losses import MultiHuberLoss

name = 'MNIST'
lr = 0.001
epochs = 201

owd_weights = [ 0, 0.001 ]

batch_size_train = 32
batch_size_test = 32

def trainloader(tsp = 100, aug=False):
    if aug:
        train_transform = transforms.Compose(
                    [
                    #transforms.ToPILImage(),
                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                    ])
    else:
        train_transform = transforms.Compose([
              transforms.ToTensor(),
              # Normalize a tensor image with mean and standard deviation
              transforms.Normalize(mean=(0.1307,), std=(0.3081,))
          ])

    training_set = datasets.MNIST(
          '../data-mnist',
          train=True,
          download=True,
          transform=train_transform)

    if tsp < 100:
        training_set = torch.utils.data.Subset(training_set, range(int(len(training_set)*tsp/100)))

    return torch.utils.data.DataLoader(
      training_set,
      batch_size=batch_size_train,
      shuffle=True)

def testloader():
    return torch.utils.data.DataLoader(
      datasets.MNIST(
          '../data',
          train=False,
          download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              # Normalize a tensor image with mean and standard deviation              
              transforms.Normalize(mean=(0.1307,), std=(0.3081,))
          ])),
      batch_size=batch_size_test,
      shuffle=True)


# LeNet - https://www.kaggle.com/usingtc/lenet-with-pytorch
class LeNet(nn.Module):

    def __init__(self, dropout = False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc1_1   = nn.Linear(120, 84, bias=False)
        self.fc2   = nn.Linear(84, 10)
        self.dropout = dropout

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, 16*5*5)

        if self.dropout:
            x = F.dropout(x, training=self.training) 
		
        x = F.relu(self.fc1(x))

        if self.dropout:
            x = F.dropout(x, training=self.training) 

        x = F.relu(self.fc1_1(x))
        z = x

        if self.dropout:
            x = F.dropout(x, training=self.training) 

        x = self.fc2(x)

        return x, z

def model(dropout = False):
    return LeNet(dropout)
