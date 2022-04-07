from src.modelBuilder import ModelBuilder


def test():

    builder = ModelBuilder()

    data = builder.load_data()

    xtrain, xtest, ytrain, ytest = builder.split_data_ann(data)

    ann_classifier, train_loss = builder.ann(xtrain, xtest, ytrain, ytest)

    builder.plotANN(train_loss)

    return None