from src.modelBuilder import ModelBuilder
from src.dataPreprocessing import DPP
from src.plotters import Plotters as PL

def test():

    builder = ModelBuilder()

    data = DPP.load_data()

    xtrain, xtest, ytrain, ytest = DPP.splitter(data)

    model = builder.dt(xtrain, xtest, ytrain, ytest)

    PL.plotTree(model)

    return None