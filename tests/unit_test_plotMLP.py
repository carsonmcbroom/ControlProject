from src.modelBuilder import ModelBuilder


def test():

    builder = ModelBuilder()

    data = builder.load_data()

    xtrain, xtest, ytrain, ytest = builder.split_data(data)

    model = builder.mlp(xtrain, xtest, ytrain, ytest)

    builder.plotLoss(model)

    return None