"""
This data contains protein localization sites
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset_peek import data_peek


def load_ecoli():
    fp = "ecoli.csv"
    raw_data = pd.read_csv(fp, delim_whitespace=True, header=None)
    x = np.array(raw_data.iloc[:, 1:8])
    y = np.array(raw_data.iloc[:, 8])
    y = LabelEncoder().fit_transform(y)
    return x, y


if __name__ == '__main__':
    data_peek("Ecoli", load_ecoli)
