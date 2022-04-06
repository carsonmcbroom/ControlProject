from src.dataPreprocessing import DPP

def test():
    preprocessor = DPP()
    data = preprocessor.load_data()
    print('test has run')
    return data
