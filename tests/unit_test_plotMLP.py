from src.modelBuilder import ModelBuilder
from src.dataPreprocessing import DPP
from src.plotters import Plotters as PL


def test():

    builder = ModelBuilder()

    data = DPP.load_data()

    xtrain, xtest, ytrain, ytest = DPP.split_data(data)

    model = builder.mlp(xtrain, xtest, ytrain, ytest)

    PL.plotLoss(model)

    return None