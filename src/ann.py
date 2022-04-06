#package import

import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class ANN():
    def __init__(self) -> None:
        pass
        
def dataloader(self, xtrain, xtest, ytrain, ytest):
    batch_size = 10

    trainset = []
    for i in range(len(xtrain)):
        trainset.append([xtrain[i], ytrain[i]])

    testset = []
    for i in range(len(xtest)):
        testset.append([xtest[i], ytest[i]])

    trainingload = DataLoader(trainset, batch_size = batch_size, shuffle = True)
    testingload = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainingload, testingload

def train(self, net, trainingload):
    train_loss = []

    criterion = nn.CrossEntropyLoss()
