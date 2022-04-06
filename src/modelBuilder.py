from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn

from src.dataPreprocessing import DPP
from src.plotters import Plotters
from src.ann import ANN, Net

class ModelBuilder(DPP, Plotters, ANN, Net):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def mlp(self, xtrain, xtest, ytrain, ytest):

        MLP_classifier = MLPClassifier(hidden_layer_sizes=(200, ), max_iter = 3000, verbose = False, learning_rate_init = 0.01, random_state = 1)

        MLP_classifier.fit(xtrain, ytrain)

        MLP_predicted = MLP_classifier.predict(xtest)

        error = 0
        for i in range(len(ytest)):
            error += np.sum(MLP_predicted != ytest)

        accuracy_tot = 1 - error / len(ytest)

        MLP_accuracy = accuracy_score(ytest, MLP_predicted)

        return MLP_classifier

    def dt(self, xtrain, xtest, ytrain, ytest):
        #Create DT model
        DT_classifier = DecisionTreeClassifier()

        #Train the model
        DT_classifier.fit(xtrain, ytrain)

        #Test the model
        DT_predicted = DT_classifier.predict(xtest)

        error = 0
        for i in range(len(ytest)):
            error += np.sum(DT_predicted != ytest)

        total_accuracy = 1 - error / len(ytest)

        DT_accuracy = accuracy_score(ytest, DT_predicted)

        #get performance
        DT_accuracy = accuracy_score(ytest, DT_predicted)

        return DT_classifier

    def ann(self, xtrain, xtest, ytrain, ytest):
        ann_classifier = ANN()
        model = Net()
        print(model)

        trainingload, testingload = ann_classifier.dataloader(
            xtrain, xtest, ytrain, ytest)

        train_loss = ann_classifier.train(model, trainingload)

        accuracy = ann_classifier.accuracy2(model, testingload)

        return ann_classifier, train_loss



