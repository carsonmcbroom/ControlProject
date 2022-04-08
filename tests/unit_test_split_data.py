from src.dataPreprocessing import DPP

def test():
    preprocessor = DPP()
    data = preprocessor.load_data()

    xtrain, xtest, ytrain, ytest = preprocessor.splitter(data)

    return None
