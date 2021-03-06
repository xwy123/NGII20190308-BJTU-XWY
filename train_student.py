import sys
sys.path.append("/mnt/SaiER/VGG")
import os
import time
import torch
import torchvision
from torchvision import transforms
import torch.utils.data
import numpy as np

from VGG import vgg11

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './CIFAR10'
NUM_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 6e-4

list_name = []
loss_value = []

model = vgg11().to(device)

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
    LossFunction = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        start = time.time()
        loss_count = 1
        loss_all = 0

        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = LossFunction(outputs, labels)
            
            loss_all += loss.item()
            loss_count = loss_count + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if loss_count == 50:
                loss_value.append(loss_all/loss_count)
                loss_all = 0
                loss_count = 0

            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(dataset)}], "
                  f"Loss: {loss.item():.8f}.")
            step += 1

        end = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
              f"time: {end - start} sec!")

    
def getName():
    for name in model.state_dict():
        list_name.append(name)

        
def saveModel():
    for name in list_name:
        temp_np = model.state_dict()[name].cpu().numpy()
        np.save("./SaveModel11_noteacher/%s.ndim" % (name), temp_np)

        
def saveLossInfo():
    with open('./SaveInfo/vgg11_noteacher_loss_t.txt','w') as fileWriter:
        for value in loss_value:
            fileWriter.write(str(value)+"-")
        fileWriter.close() 
    

if __name__ == '__main__':
    print(model)
    getName()
    train_net()
    saveModel()
    saveLossInfo()
