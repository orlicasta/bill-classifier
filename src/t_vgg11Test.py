import os
#import random
import csv
#import pandas as pd
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import torch
#import torchvision
#from showImages import imshow
from statistics import mean
#import torch.nn as nn
import torchvision.transforms as transforms
#from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.utils.data.dataset import Dataset
#import itertools
#import time
import shutil
#import timm
#from datetime import datetime
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassF1Score
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report

from customDatasetFromImages import CustomDatasetFromImages
#from cnn import CNN

if __name__ == '__main__':
	testData = CustomDatasetFromImages("test.csv", os.getcwd() ,transform = transforms.ToTensor())

modelName = "vgg11Model"
net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)

backupDir = os.getcwd() + "/results/" + modelName
runNum = os.listdir(backupDir).__len__()
runNum = runNum - 1

PATH = backupDir + "/" + str(runNum) + "/" + modelName + ".pth"

batch_size = 1
test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)

net.load_state_dict(torch.load(PATH))
net.eval()

classes = ['other', 'atmos', 'att', 'capitalone', 'kohl']

resultsFile = open("results.csv", 'w', newline='')
wrtr = csv.writer(resultsFile)
wrtr.writerow(["groundtruth","activationratio", "prediction", classes[0], classes[1], classes[2], classes[3], classes[4]])

with torch.no_grad():
    correct = 0
    total = 0
    actiratios = []
    successRatios = []
    failRatios = []
    targets = []
    preds = []

    for images, labels in test_loader:
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        
        print("GroundTruth:", classes[labels.item()])
        #print("top neuron activation strength", _)
        print("Prediction:", classes[predicted.item()])
        aratio = _.item() / (outputs[0][0].item() + outputs[0][1].item() + outputs[0][2].item() + outputs[0][3].item() + outputs[0][4].item())
        print("Activation Ratio:", aratio, "\n")
        #print("neuron activations:", outputs[0][0].item(), outputs[0][1].item(), outputs[0][2].item(), outputs[0][3].item(), outputs[0][4].item(), "\n")
        
        wrtr.writerow([classes[labels.item()], str(aratio), classes[predicted.item()], outputs[0][0].item(), outputs[0][1].item(), outputs[0][2].item(), outputs[0][3].item(), outputs[0][4].item()])

        actiratios.append(aratio)

        #for precision metric
        targets.append(labels.item())
        preds.append(np.array(outputs[0][0:5]))

        #if(total > 100):

            #targets = np.array(targets)
            #print(targets)
            #targets = torch.Tensor(targets)
            #print(targets)
            #preds = np.array(preds)
            #preds = torch.Tensor(preds)

            #print(classes)

            #metric = MulticlassPrecision(num_classes=5)
            #print(metric(preds, targets))
        
        if predicted == labels:
            successRatios.append(aratio)
        else:
            failRatios.append(aratio)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    resultsFile.close()




    shutil.copy("results.csv", backupDir + "/" + str(runNum) + "/results.csv")

    print('Accuracy: {} %'.format(100 * correct / total))
    print("")
    print("Average activation ratio:", mean(actiratios))
    print("Average ratio for successful predictions:", mean(successRatios))
    print("Average ratio for failed predictions:", mean(failRatios))
    print("")

    targets = np.array(targets)
    preds = np.array(preds)
    targets = torch.Tensor(targets)
    preds = torch.Tensor(preds)
    
    print("Classes ", classes)
    mcp = MulticlassPrecision(num_classes=classes.__len__(), average=None)
    print("Class Precision: ",  mcp(preds, targets))

    mcr = MulticlassRecall(num_classes=classes.__len__(), average=None)
    print("Class Recall: ", mcr(preds, targets))

    mcf1s = MulticlassF1Score(num_classes=classes.__len__(), average=None)
    print("F1 Scores: ", mcf1s(preds, targets))

    with open(backupDir + "/" + str(runNum) + "/" + str(100 * correct / total) + ".txt", 'w') as file:
        file.write('Accuracy: {} %'.format(100 * correct / total))
        file.write("\nAverage activation ratio: {}".format(mean(actiratios)))
        file.write("\nAverage ratio for successful predictions: {}".format(mean(successRatios)))
        file.write("\nAverage ratio for failed predictions: {}".format(mean(failRatios)))
        file.write("\nClass Precision: {}".format(mcp(preds, targets)))
        file.write("\nClass Recall: {}".format( mcr(preds, targets)))
        file.write("\nF1 Scores: {}".format(mcf1s(preds, targets)))

    
