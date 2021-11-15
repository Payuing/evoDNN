"""
This data set contains 10 variables that are age, gender, total Bilirubin, direct Bilirubin, total proteins, albumin, A/G ratio, SGPT, SGOT and Alkphos.
"""

import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_ilpd():
    fp = 'ilpd.csv'
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 0])
    x1 = np.array(raw_data.iloc[:, 1])
    x2 = np.array(raw_data.iloc[:, 2:10])
    x1 = LabelEncoder().fit_transform(x1)
    x = np.c_[x, x1, x2]
    y = np.array(raw_data.iloc[:, 10])
    return x, y

if __name__ == '__main__':
    data, target = load_ilpd()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
