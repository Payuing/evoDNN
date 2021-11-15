"""
Load the yeast dataset from UCI ML repository
Return numpy ndarry of data and target.
Sequence Name does not used as attribute.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_yeast():
    fp = "yeast.data.txt"
    raw_data = pd.read_csv(fp, delim_whitespace=True, header=None)
    x = np.array(raw_data.iloc[:,1:5])
    x1 = np.array(raw_data.iloc[:,7:9])
    data = np.c_[x, x1]
    y = np.array(raw_data.iloc[:,9])
    target = LabelEncoder().fit_transform(y)

    return data, target

if __name__ == '__main__':
    data, target = load_yeast()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
