import sys
sys.path.append("/file/VGG")
import os
import time
import torch
import torchvision
from torchvision import utils
from torchvision import transforms
import torch.utils.data
import numpy as np

from VGG import vgg11
from VGG import vgg19_bn
from VGG import vgg11E1
from VGG import vgg11E2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


WORK_DIR = './CIFAR10'
BATCH_SIZE = 10

list_name = []

model = vgg11E1().to(device)

transform = transforms.Compose([
    transforms.RandomCrop(36, padding=4),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


dataset = torchvision.datasets.CIFAR10(root=WORK_DIR,
                                        download=True,
                                        train=False,
                                        transform=transform)

dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

def test():
    correct = 0.
    total = 0
    num = 1
    with torch.no_grad():
        start = time.time()
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        end = time.time()
        print(f"Acc: {correct / total:.4f}.")
        print(f"time: {end - start} sec!")

        
def getName():
    for name in model.state_dict():
        list_name.append(name)

    
def loadModel():
    for var_name in list_name:
        temp_load_numpy = np.load("./SaveModel11E1/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        model.state_dict()[var_name].copy_(tensor_load)
        
if __name__=="__main__":
    getName()
    loadModel()
    test()