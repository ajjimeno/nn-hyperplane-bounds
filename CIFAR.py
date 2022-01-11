import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

name = 'CIFAR'
lr = 0.0001
epochs = 201

owd_weights = [ 0.0, 0.00001 ]

batch_size_train = 32
batch_size_test = 32

def trainloader(tsp = 100, aug = False):
    if aug:
        train_transform = transforms.Compose(
                    [
                    #transforms.ToPILImage(),
                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    else:
        train_transform = transform


    training_set = torchvision.datasets.CIFAR10(root='./data-cifar', train=True,
                                        download=True, transform=train_transform)

    if tsp < 100:
        training_set = torch.utils.data.Subset(training_set, range(int(len(training_set)*tsp/100)))

    return torch.utils.data.DataLoader(training_set, batch_size=batch_size_train,
                                          shuffle=True, num_workers=2)

def testloader():
        testset = torchvision.datasets.CIFAR10(root='./data-cifar', train=False,
                                       download=True, transform=transform)
        return torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class VGG19(nn.Module):
    def __init__(self, dropout=False):
        super(VGG19, self).__init__()
        self.vgg19 = models.vgg19(num_classes=10)
        self.dropout = dropout
        self.fc2 = self.vgg19.classifier[6]

    def forward(self, x):
        x = self.vgg19.features(x)
        x = self.vgg19.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.vgg19.classifier[0](x)
        x = self.vgg19.classifier[1](x)

        if self.dropout:
            x = self.vgg19.classifier[2](x)

        x = self.vgg19.classifier[3](x)
        x = self.vgg19.classifier[4](x)

        z = x

        if self.dropout:
            x = self.vgg19.classifier[5](x)

        x = self.fc2(x)

        return x, z

def model(dropout = False):
    return VGG19(dropout)
