from src.modelBuilder import ModelBuilder
from src.dataPreprocessing import DPP
from src.plotters import Plotters as PL

def test():

    builder = ModelBuilder()

    data = DPP.load_data()

    xtrain, xtest, ytrain, ytest = DPP.split_data_ann(data)

    ann_classifier, train_loss = builder.ann(xtrain, xtest, ytrain, ytest)

    PL.plotANN(train_loss)

    return None