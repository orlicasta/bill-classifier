import os
import sys
#import random
#import csv
#import pandas as pd
#import numpy as np
#import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
#from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.utils.data.dataset import Dataset
#import timm
import time
import shutil
from datetime import datetime
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from customDatasetFromImages import CustomDatasetFromImages

if __name__ == '__main__':
	train_set = CustomDatasetFromImages("train.csv", os.getcwd() ,transform = transforms.ToTensor())

num_epochs = int(sys.argv[1])
batch_size = 20
learning_rate = 0.0003

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

modelName = "alexModel"

weights = torchvision.models.AlexNet_Weights.DEFAULT
auto_transforms = weights.transforms()
print(auto_transforms)

net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
print(net)


#freeze feature-learning layers
#for param in net.features.parameters():
#    param.requires_grad = False


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

start_time = time.time()

for epoch in range(num_epochs):
    print ("Starting Epoch {}".format(epoch + 1))

    train_iter = iter(train_loader)

    losses = []
    i = 0
    
    for data, targets in train_iter: 

        outputs = net(data.float())
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        i += 1
        if (i) % batch_size == 0:
            print('Epoch {}/{}, Iter {}/{}, Loss: {}'.format(epoch + 1, num_epochs, (i / batch_size).__int__(), ((train_set.__len__() / batch_size) / batch_size).__int__(), loss.item()))
        
    print("Epoch Done")

print("--- %s seconds ---" % (time.time() - start_time))

backupDir = os.getcwd() + "/results/" + modelName
dirExist = os.path.exists(backupDir)
if not dirExist:
    os.makedirs(backupDir)

runNum = os.listdir(backupDir).__len__()
os.makedirs(backupDir + "/" + str(runNum))

d = datetime.now()
d = d.strftime('_at_%Y-%m-%d_%H-%M-%S')

with open(backupDir + "/" + str(runNum) + "/" + str(num_epochs) + "_epochs_" + str(learning_rate) + "_lr_for_" + str(time.time() - start_time) + d + ".txt", 'w') as file:
    file.write(d)

shutil.copy("train.csv", backupDir + "/" + str(runNum) + "/train.csv")
shutil.copy("test.csv", backupDir + "/" + str(runNum) + "/test.csv")

PATH = backupDir + "/" + str(runNum) + "/" + modelName + ".pth"
torch.save(net.state_dict(), PATH)