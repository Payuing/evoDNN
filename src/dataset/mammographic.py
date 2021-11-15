"""
Discrimination of benign and malignant mammographic masses based on BI-RADS attributes and the patient's age.
"""
import numpy as np, pandas as pd

def load_mammographic():
    fp = 'mammographic_masses.csv'
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:5])
    y = np.array(raw_data.iloc[:, 5])
    return x, y


if __name__ == '__main__':
    data, target = load_mammographic()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
