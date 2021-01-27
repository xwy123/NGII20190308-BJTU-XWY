import sys
sys.path.append("/mnt/SaiER/VGG")
import os
import time
import torch
import torchvision
from torchvision import transforms
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from VGG import vgg11
from VGG import vgg11E1
from VGG import vgg11E2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './CIFAR10'
NUM_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 6e-4

list_name = []
list_name_E = []
loss_value = []

net = vgg11()
netE1 = vgg11E1()
netE2 = vgg11E2()
net.to(device)
netE1.to(device)
netE2.to(device)

transform = transforms.Compose([
    transforms.RandomCrop(36, padding=4),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


dataset = torchvision.datasets.CIFAR10(root=WORK_DIR,
                                        download=True,
                                        train=True,
                                        transform=transform)

dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

dataset_test = torchvision.datasets.CIFAR10(root=WORK_DIR,
                                        download=True,
                                        train=False,
                                        transform=transform)

dataset_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)


def train_net():
    
    print(f"Train numbers:{len(dataset)}")
    cast = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, netE1.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-8)
    step = 1
    
    for epoch in range(1, NUM_EPOCHS + 1):
        netE1.train()
        start = time.time()

        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = netE1(images)
            loss = cast(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(dataset)}], "
                  f"Loss: {loss.item():.8f}.")
            step += 1

        end = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
              f"time: {end - start} sec!")


def getName():
    for name in net.state_dict():
        list_name.append(name)

def getNameE():
    for name in netE1.state_dict():
        list_name_E.append(name)


def saveModel():
    for name in list_name_E:
        temp_np = netE1.state_dict()[name].cpu().numpy()
        np.save("./SaveModel11E1/%s.ndim" % (name), temp_np)


def loadModel():
    for var_name in list_name:
        temp_load_numpy = np.load("./SaveModel11_teacherT3A08/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        net.state_dict()[var_name].copy_(tensor_load)

def copyWeight():
    for i in range(6):
        netE1.state_dict()[list_name_E[i]].copy_(net.state_dict()[list_name[i]])

def seeE1_data():
    for j in range(6):
        print(netE1.state_dict()[list_name_E[j]])
        

def see_data():
    for j in range(6):
        print(net.state_dict()[list_name[j]])
        

def frozenPara():
    for i, p in enumerate(netE1.parameters()):
        if i < 6:
            p.requires_grad = False
        print(p.requires_grad)
        
        
if __name__ == "__main__":
    getName()
    getNameE()
    loadModel()
    copyWeight()
    frozenPara()
    train_net()
    seeE1_data()
    saveModel()
    