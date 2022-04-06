from src.modelBuilder import ModelBuilder


def test():

    builder = ModelBuilder()

    data = builder.load_data()

    X_train, X_test, y_train, y_test = builder.split_data_ann(data)

    ann_classifier, train_loss = builder.ann(X_train, X_test, y_train, y_test)

    builder.plotANN(train_loss)

    return None