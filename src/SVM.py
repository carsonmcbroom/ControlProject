
from sklearn.svm import SVC
from sklearn.metrics import classification_report as CLR
from sklearn.metrics import confusion_matrix as CMX

from dataPreprocessing import DPP


#make function w/ xtrain, ytrain and unit test

#Load Data
preprocessor = DPP()
data = preprocessor.load_data()
xtrain, xtest, ytrain, ytest = preprocessor.splitter(data)
ytrain = ytrain.values.ravel()

#Classifier
clsf = SVC(kernel='linear')
clsf.fit(xtrain, ytrain) 

#Prediction
y_pred = clsf.predict(xtest)

#Evaluation
print(CMX(ytest,y_pred))
print(CLR(ytest,y_pred))
