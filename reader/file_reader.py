import pandas as pd

def read_file(path):
    splittedPath = path.split('.')
    file_xtension = splittedPath[-1]
    if(file_xtension == '.csv'):
        df = pd.read_csv(path)
    if(file_xtension == '.csv'):
        df = pd.read_excel(path)
    return df