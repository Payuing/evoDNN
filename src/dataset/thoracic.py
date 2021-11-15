"""
he data is dedicated to classification problem related to the post-operative life expectancy in the lung cancer patients: class 1 - death within one year after surgery, class 2 - survival.
"""

import numpy as np, pandas as pd

def load_thoracic():
    fp = "thoracic.csv"
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:,0:16])
    y = np.array(raw_data.iloc[:, 16])
    return x, y

if __name__ == '__main__':
    data, target = load_thoracic()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
