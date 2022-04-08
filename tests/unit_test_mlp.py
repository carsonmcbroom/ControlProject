from src.modelBuilder import ModelBuilder
from src.dataPreprocessing import DPP
def mlptest():
    builder = ModelBuilder()

    data = DPP.load_data()

    xtrain, xtest, ytrain, ytest = DPP.splitter(data)

    model = builder.ann(xtrain, xtest, ytrain, ytest)

    return None
