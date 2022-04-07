from src.modelBuilder import ModelBuilder

def mlptest():
    builder = ModelBuilder()

    data = builder.load_data()

    xtrain, xtest, ytrain, ytest = builder.split_data(data)

    model = builder.ann(xtrain, xtest, ytrain, ytest)

    return None
