
from sklearn.svm import SVC
from sklearn.metrics import classification_report as CLR
from sklearn.metrics import confusion_matrix as CMX
from torch.utils.data import DataLoader

from dataPrepocessing import load_data, splitter

#make function w/ xtrain, ytrain and unit test

class SVM():
    def __init__(self) -> None:
        pass

    def dataloader(self, X_train, X_test, y_train, y_test):
        batch_size = 10

        trainset = []
        for i in range(len(X_train)):
            trainset.append([X_train[i], y_train[i]])

        testset = []
        for i in range(len(X_test)):
            testset.append([X_test[i], y_test[i]])

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        return trainloader, testloader

    def SVM_Classifier(xtrain, ytrain, xtest, ytest):

        #Classifier
        clsf = SVC(kernel='linear')
        clsf.fit(xtrain, ytrain) 

        #Prediction
        y_pred = clsf.fit(xtest)

        #Evaluation
        print(CMX(ytest,y_pred))
        print(CLR(ytest,y_pred))

        return(None)


