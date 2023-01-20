# imports for the tutorial
import numpy as np
import matplotlib.pyplot as plt
import time
import os, datetime
import copy
import itertools
import optuna
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset, RandomSampler
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
import torchvision
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
import pandas as pd
import seaborn as sn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
varImageNet = [0.229, 0.224, 0.225]
invVar = [1/0.229, 1/0.224, 1/0.225]
meanImageNet = [0.485, 0.456, 0.406]
invMean = [-0.485,- 0.456, -0.406]

plt.rc('xtick', labelsize=26)
plt.rc('ytick', labelsize=26)
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2

cos = nn.CosineSimilarity()

# function to calcualte accuracy of the model
def calculate_accuracy(args, model_list, dataloader):
    for model in model_list:
        model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([args.num_classes,args.num_classes], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            for i, model in enumerate(model_list):
                outputs = model(images)
                if i == 0:
                    total_output = outputs/len(model_list)
                else:
                    total_output += outputs/len(model_list)

            _, predicted = torch.max(total_output.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

