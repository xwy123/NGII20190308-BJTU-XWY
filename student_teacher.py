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
from VGG import vgg19_bn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './CIFAR10'
NUM_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 6e-4
alpha = 0.8

list_name = []
teacher_list_name = []
loss_value = []

student_model = vgg11().to(device)
teacher_model = vgg19_bn().to(device)

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
    KLDill = nn.KLDivLoss()
    # Optimization
    optimizer = torch.optim.Adam(
        student_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)
    step = 1

    for epoch in range(1, NUM_EPOCHS + 1):
        student_model.train()

        start = time.time()
        loss_count = 1
        loss_all = 0

        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = student_model(images)
            loss1 = cast(outputs, labels)

            optimizer.zero_grad()

            soft_target = teacher_model(images)

            Temperature = 5
            output_S = F.log_softmax(outputs/Temperature, dim=1)
            output_T = F.softmax(soft_target/Temperature, dim=1)

            loss2 = KLDill(output_S, output_T) * Temperature * Temperature

            loss = loss1*(1-alpha) + loss2*alpha

            loss_all += loss.item()
            loss_count = loss_count + 1

            loss.backward()
            optimizer.step()

            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(dataset)}], "
                  f"Loss: {loss.item():.8f}.")

            if loss_count == 50:
                loss_value.append(loss_all/loss_count)
                loss_all = 0
                loss_count = 0

            step += 1

        end = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
              f"time: {end - start} sec!")


    
def getName():
    for name in student_model.state_dict():
        list_name.append(name)


def getTeaName():
    for name in teacher_model.state_dict():
        teacher_list_name.append(name)

        
def saveModel():
    for name in list_name:
        temp_np = student_model.state_dict()[name].cpu().numpy()
        np.save("./SaveModel11_teacherT5A03/%s.ndim" % (name), temp_np)


def saveLossInfo():
    with open('./SaveInfo/vgg11_teacher_loss_T5A03.txt','w') as fileWriter:
        for value in loss_value:
            fileWriter.write(str(value)+"-")
        fileWriter.close() 
        
        
def loadModel():
    print("Load teacher Model...........")
    for var_name in teacher_list_name:
        temp_load_numpy = np.load("./SaveModel19_bn_lr6e4/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        teacher_model.state_dict()[var_name].copy_(tensor_load)

        
if __name__ == '__main__':
    print(student_model)
    getName()
    getTeaName()
    loadModel()
    train_net()
    saveModel()
    saveLossInfo()