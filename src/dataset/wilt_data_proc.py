"""
Load the wilt data set detecting diseased trees
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_wilt():
    fp_train = "wilt_training.csv"
    fp_test = "wilt_testing.csv"

    raw_data_train = pd.read_csv(fp_train, delimiter=',')
    raw_data_test = pd.read_csv(fp_test, delimiter=',')
    frames = [raw_data_train, raw_data_test]
    raw_data = pd.concat(frames)
    data = np.array(raw_data.iloc[:,1:6])
    y = np.array(raw_data.iloc[:,0])
    target = LabelEncoder().fit_transform(y)
    return data, target

if __name__ == '__main__':
    data, target = load_wilt()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
