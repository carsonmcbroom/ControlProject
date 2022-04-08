from src.dataPreprocessing import DPP

def loadtest():
    preprocessor = DPP()
    data = preprocessor.load_data()
    print(data)
    return data
