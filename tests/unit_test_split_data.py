from src.dataPreprocessing import DPP

def test():
    preprocessor = DPP()
    data = preprocessor.load_data()

    xtrain, xtest, ytrain, ytest = preprocessor.split_data(data)

    return None
