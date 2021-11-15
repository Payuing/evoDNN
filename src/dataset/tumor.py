"""
This is one of three domains provided by the Oncology Institute
     that has repeatedly appeared in the machine learning literature.
"""
import numpy as np, pandas as pd

def load_tumor():
    fp = "tumor.csv"
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:3])
    x1 = np.array(raw_data.iloc[:, 5:18])
    y = np.array(raw_data.iloc[:, 0])
    x = np.c_[x, x1]
    return x, y

if __name__ == '__main__':
    data, target = load_tumor()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
