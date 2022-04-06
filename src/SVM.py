
from sklearn.svm import SVC
from sklearn.metrics import classification_report as CLR
from sklearn.metrics import confusion_matrix as CMX

#make function w/ xtrain, ytrain and unit test

#Classifier
clsf = SVC()
clsf.fit(xtrain, ytrain) 

#Prediction
y_pred=clsf.fit(xtest)

#Evaluation
print(CMX(ytest,y_pred))
print(CLR(ytest,y_pred))
