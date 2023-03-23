import os
import sys
#import random
#import csv
#import pandas as pd
#import numpy as np
#import cv2
#import matplotlib.pyplot as plt
import torch
#import torchvision
#from statistics import mean
#import torch.nn as nn
import torchvision.transforms as transforms
#from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.utils.data.dataset import Dataset
#import itertools
#import time
#import shutil
#import timm
#from datetime import datetime
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report

from ds4one import CustomDatasetFromImages

if __name__ == '__main__':
	testData = CustomDatasetFromImages(sys.argv[1], sys.argv[2], os.getcwd() ,transform = transforms.ToTensor())


#modelName = "vgg11Model"
modelName = "res18Model"
#net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11')
net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')

PATH = os.getcwd() + "/src/standalone/" + modelName + ".pth"

test_loader = DataLoader(testData, batch_size=1, shuffle=False)

net.load_state_dict(torch.load(PATH))
net.eval()

classes = ['other', 'atmos', 'att', 'capitalone', 'kohl']

with torch.no_grad():
    correct = 0
    total = 0
    actiratios = []
    successRatios = []
    failRatios = []
    for images, labels in test_loader:
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        print("Input image:", sys.argv[1], '<br><br><br>')
        print("ResNet18 Output<br>")

        otherratio = outputs[0][0].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        atmosratio = outputs[0][1].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        attratio = outputs[0][2].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        capitaloneratio = outputs[0][3].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        kohlratio = outputs[0][4].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())

        print('<br>', 'other:', '{:.4f}'.format(otherratio) , '<br>')
        print('atmos:', '{:.4f}'.format(atmosratio) , '<br>')
        print('att:', '{:.4f}'.format(attratio) , '<br>')
        print('capitalone:', '{:.4f}'.format(capitaloneratio) , '<br>')
        print('kohl:', '{:.4f}'.format(kohlratio) , '<br>')
        print('<br>', 'prediction: ', classes[predicted.item()], '<br><br>')

modelName = "vgg11Model"
net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11')

PATH = os.getcwd() + "/src/standalone/" + modelName + ".pth"

test_loader = DataLoader(testData, batch_size=1, shuffle=False)

net.load_state_dict(torch.load(PATH))
net.eval()

classes = ['other', 'atmos', 'att', 'capitalone', 'kohl']

with torch.no_grad():
    correct = 0
    total = 0
    actiratios = []
    successRatios = []
    failRatios = []
    for images, labels in test_loader:
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        print("<br><br>VGG11 Output<br>")

        otherratio = outputs[0][0].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        atmosratio = outputs[0][1].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        attratio = outputs[0][2].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        capitaloneratio = outputs[0][3].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        kohlratio = outputs[0][4].item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())

        print('<br>', 'other:', '{:.4f}'.format(otherratio) , '<br>')
        print('atmos:', '{:.4f}'.format(atmosratio) , '<br>')
        print('att:', '{:.4f}'.format(attratio) , '<br>')
        print('capitalone:', '{:.4f}'.format(capitaloneratio) , '<br>')
        print('kohl:', '{:.4f}'.format(kohlratio) , '<br>')
        print('<br>', 'prediction: ', classes[predicted.item()], '<br><br><br><br>')

