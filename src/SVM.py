from sklearn.svm import SVC
from sklearn.metrics import classification_report as CLR
from sklearn.metrics import confusion_matrix as CMX
from dataPreprocessing import DPP

#Load Data
preprocessor = DPP()
data = preprocessor.load_data()
xtrain, xtest, ytrain, ytest = preprocessor.splitter(data)

ytrain = ytrain.values.ravel()   #Used ravel on ytrain since SVC().fit needs 1D array for y input 

#Classifier
clsf = SVC()   #used default kernel (rbf) for better fit
clsf.fit(xtrain, ytrain) 

#Prediction
y_pred = clsf.predict(xtest)

#Evaluation
print(CMX(ytest,y_pred))
print(CLR(ytest,y_pred))