"""
Load the abalone dataset from UCI ML repository
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_abalone():
    fp = "abalone.csv"
    raw_data = pd.read_csv(fp, delimiter=',', header=None)
    x = np.array(raw_data.iloc[:,0])
    x2 = np.array(raw_data.iloc[:,1:8])
    x = LabelEncoder().fit_transform(x)
    data = np.c_[x, x2]
    target = np.array(raw_data.iloc[:,8])

    return data, target

if __name__ == '__main__':
    data, target = load_abalone()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
