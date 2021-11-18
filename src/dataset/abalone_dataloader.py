import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset_peek import data_peek


def load_abalone():
    fp = "abalone.csv"
    raw_data = pd.read_csv(fp, delimiter=',', header=None)
    x = np.array(raw_data.iloc[:, 0])
    x2 = np.array(raw_data.iloc[:, 1:8])
    x = LabelEncoder().fit_transform(x)
    x = np.c_[x, x2]
    y = np.array(raw_data.iloc[:, 8])
    return x, y


if __name__ == '__main__':
    data_peek("Abalone", load_abalone)
