#Importing packages
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

class DPP():
    def __init__(self) -> None:
        pass

def load_data(self):
    #loading data
    x_data = pd.read_csv(r'https://raw.githubusercontent.com/carsonmcbroom/ControlProject/main/RawData/RawData.csv')
    y_data = pd.read_csv(r'https://raw.githubusercontent.com/carsonmcbroom/ControlProject/main/RawData/RawData_Label.csv')

    return x_data, y_data

def splitter(self, x_data, y_data):

    x = x_data.drop(columns = "Index")
    y = y_data['label']

    scaler = StandardScaler()

    x = scaler.fit_transform(x)

    xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.2, random_state = 42)

    return xtrain, xtest, ytrain, ytest

def split_data_ann(self, x_data, y_data):
    x = x_data.drop(columns = "Index")
    y = y_data['label']

    scaler = StandardScaler()
    x =scaler.fit_transform(x)

    xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.2, random_state = 42)

    return xtrain, xtest, ytrain, ytest
