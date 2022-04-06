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
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    epochs = 50

    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainingload, 0):
            x, labels = data
            
            optimizer.zero_grad()

            outputs = net(x.float())

            loss = criterion(outputs, labels.long())

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        loss = running_loss / len(trainingload)
        train_loss.append(loss)

        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch + 1, epochs, loss))

    return train_loss

def accuracy(self, model, testingload):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for x, labels in testingload:
            outputs = model(x.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()

        percentage = (correct / total) * 100

        print('Test Accuracy of the model: {} %'.format(
                percentage))
    return percentage

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Traditional NNs (fully connected layers)
        self.fc0 = nn.Linear(in_features=36, out_features=200)
        self.fc1 = nn.Linear(in_features=200, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

