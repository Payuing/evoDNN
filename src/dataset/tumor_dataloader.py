"""
This is one of three domains provided by the Oncology Institute
     that has repeatedly appeared in the machine learning literature.
"""
import numpy as np
import pandas as pd
from dataset_peek import data_peek


def load_tumor():
    fp = "tumor.csv"
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:3])
    x1 = np.array(raw_data.iloc[:, 5:18])
    x = np.c_[x, x1]
    y = np.array(raw_data.iloc[:, 0])
    return x, y


if __name__ == '__main__':
    data_peek("Tumor", load_tumor)
