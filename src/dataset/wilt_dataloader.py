"""
Load the wilt data set detecting diseased trees
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset_peek import data_peek


def load_wilt():
    fp_train = "wilt_training.csv"
    fp_test = "wilt_testing.csv"
    raw_data_train = pd.read_csv(fp_train, delimiter=',')
    raw_data_test = pd.read_csv(fp_test, delimiter=',')
    frames = [raw_data_train, raw_data_test]
    raw_data = pd.concat(frames)
    x = np.array(raw_data.iloc[:, 1:6])
    y = np.array(raw_data.iloc[:, 0])
    y = LabelEncoder().fit_transform(y)
    return x, y


if __name__ == '__main__':
    data_peek("Wilt", load_wilt)
