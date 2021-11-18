"""
Data on cardiac Single Proton Emission Computed Tomography (SPECT) images. Each patient classified into two categories: normal and abnormal.
"""
import numpy as np
import pandas as pd
from dataset_peek import data_peek


def load_spectf():
    fp_test = "spectf_test.csv"
    fp_train = "spectf_train.csv"
    raw_data_test = pd.read_csv(fp_test, header=None)
    raw_data_train = pd.read_csv(fp_train, header=None)
    frames = [raw_data_train, raw_data_test]
    raw_data = pd.concat(frames)
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(raw_data.iloc[:, 0])
    return x, y


if __name__ == '__main__':
    data_peek("Spectf", load_spectf)
