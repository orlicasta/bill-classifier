#import os
import pandas as pd
import numpy as np
#import cv2
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
#import sys
from PIL import Image
#from skimage import io
#from skimage.transform import resize
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report


class CustomDatasetFromImages(Dataset):
    def __init__(self, csvFile, rootDir, transform=None):
        self.annotations = pd.read_csv(csvFile)

        print(self.annotations)

        self.image_array = np.asarray(self.annotations.iloc[:, 0])
        self.label_array = np.asarray(self.annotations.iloc[:, 1])

        # this might be changed to the transform in __getitem__ method
        self.transform = transform

        # width and height for image transform
        self.tDim = (400, 400)

        self.rootDir = rootDir

    def __getitem__(self, index):
        #img_path = os.path.join(self.rootDir, self.annotations.iloc[index, 0])
        #image = io.imread(img_path)

        # get image path
        single_image_name = self.image_array[index]
        #image_as_image = cv2.imread(single_image_name, 0)

        #print("looking at:", single_image_name)

        # open image
        # not all images are jpg, so convert to 3-channel image
        myImage = Image.open(single_image_name).convert('RGB')
        # resize open image
        myImage = myImage.resize(self.tDim)

        # tranform resized pil image to tensor
        myTransform = transforms.Compose([transforms.PILToTensor()])
        img_tensor = myTransform(myImage)

        # transfrom label column to tensor
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        #if self.transform:
            #image_final = self.transform(image_final)

        return (img_tensor, label)

    def __len__(self):
        return len(self.annotations)