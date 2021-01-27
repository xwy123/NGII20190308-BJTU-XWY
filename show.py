import sys
sys.path.append("../SaierProject")
import os
import time
import torch
import torchvision
from torchvision import transforms
import torch.utils.data
import numpy as np
import random
from torchvision import utils
from random import choice
import argparse

from VGG import vgg11
from VGG import vgg11E1
from VGG import vgg11E2

parser = argparse.ArgumentParser(description='Model Exits Selection')
parser.add_argument("--ImageType", type=str, default='cat')
parser.add_argument("--SpeedSelect", type=str, default='MidHigh')

args = parser.parse_args()

image_type = args.ImageType
speed_select = args.SpeedSelect

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

WORK_DIR = './CIFAR10'
BATCH_SIZE = 1

list_nameE1 = []
list_nameE2 = []
list_nameE3 = []

modelE1 = vgg11E1()
modelE2 = vgg11E2()
modelE3 = vgg11()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelE1.to(device)
modelE2.to(device)
modelE3.to(device)

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


def testE1():
    correct = 0
    c_sum =0
    with torch.no_grad():
        start = time.time()
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = modelE1(images)
            _, predicted = torch.max(outputs.data, 1)
            if classification[predicted.cpu().numpy()[0]] == image_type:
                utils.save_image(images,
                                 './SaveResultImage/' + classification[predicted.cpu().numpy()[0]] + str(correct)
                                 + '.jpg', normalize=True)
                correct += 1
                if predicted == labels:
                    c_sum += 1
        end = time.time()
        print("已筛选总图片数：" + str(correct))
        print("预测分类正确率：" + str((c_sum/correct)*100)+" %")
        print(f"time: {end - start:.4f} sec!")


def testE2():
    correct = 0
    c_sum = 0
    with torch.no_grad():
        start = time.time()
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = modelE2(images)
            _, predicted = torch.max(outputs.data, 1)
            if classification[predicted.cpu().numpy()[0]] == image_type:
                utils.save_image(images,
                                 './SaveResultImage/' + classification[predicted.cpu().numpy()[0]] + str(correct)
                                 + '.jpg', normalize=True)
                correct += 1
                if predicted == labels:
                    c_sum += 1
        end = time.time()
        print("已筛选总图片数：" + str(correct))
        print("预测分类正确率：" + str((c_sum / correct) * 100) + " %")
        print(f"time: {end - start:.4f} sec!")


def testE3():
    correct = 0
    c_sum = 0
    with torch.no_grad():
        start = time.time()
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = modelE3(images)
            _, predicted = torch.max(outputs.data, 1)
            if classification[predicted.cpu().numpy()[0]] == image_type:
                utils.save_image(images,
                                 './SaveResultImage/' + classification[predicted.cpu().numpy()[0]] + str(correct)
                                 + '.jpg', normalize=True)
                correct += 1
                if predicted == labels:
                    c_sum += 1
        end = time.time()
        print("已筛选总图片数：" + str(correct))
        print("预测分类正确率：" + str((c_sum / correct) * 100) + " %")
        print(f"time: {end - start:.4f} sec!")


def getNameE1():
    for name in modelE1.state_dict():
        list_nameE1.append(name)


def getNameE2():
    for name in modelE2.state_dict():
        list_nameE2.append(name)


def getNameE3():
    for name in modelE3.state_dict():
        list_nameE3.append(name)


def loadModelE1():
    for var_name in list_nameE1:
        temp_load_numpy = np.load("./SaveModel11E1/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        modelE1.state_dict()[var_name].copy_(tensor_load)


def loadModelE2():
    for var_name in list_nameE2:
        temp_load_numpy = np.load("./SaveModel11E2/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        modelE2.state_dict()[var_name].copy_(tensor_load)


def loadModelE3():
    for var_name in list_nameE3:
        temp_load_numpy = np.load("./SaveModel11_teacherT3A08/%s.ndim.npy" % (var_name))
        tensor_load = torch.tensor(temp_load_numpy)
        modelE3.state_dict()[var_name].copy_(tensor_load)


if __name__ == "__main__":
    if speed_select == 'High':
        print("Loading model1 to select images...")
        getNameE1()
        loadModelE1()
        testE1()
    elif speed_select == 'MidHigh':
        print("Loading model2 to select images...")
        getNameE2()
        loadModelE2()
        testE2()
    else:
        print("Loading model3 to select images...")
        getNameE3()
        loadModelE3()
        testE3()

