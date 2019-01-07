# -*- coding: utf-8 -*-
"""
Created on Mon Jan 07 17:22:33 2019

@author: abhishek
"""
import pandas as pd

def read_file(path):
    splittedPath = path.split('.')
    file_xtension = splittedPath[-1]
    if(file_xtension == '.csv'):
        df = pd.read_csv(path)
    if(file_xtension == '.csv'):
        df = pd.read_excel(path)
    return df
