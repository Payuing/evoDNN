"""
Load the yeast dataset from UCI ML repository
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset_peek import data_peek


def load_yeast():
    fp = "yeast.data.txt"
    raw_data = pd.read_csv(fp, delim_whitespace=True, header=None)
    x = np.array(raw_data.iloc[:, 1:5])
    x1 = np.array(raw_data.iloc[:, 7:9])
    x = np.c_[x, x1]
    y = np.array(raw_data.iloc[:, 9])
    y = LabelEncoder().fit_transform(y)
    return x, y


if __name__ == '__main__':
    data_peek("Yeast", load_yeast)
