"""
This dataset is a heart disease database similar to a database already present in the repository (Heart Disease databases) but in a slightly different form.
"""

import numpy as np, pandas as pd

def load_statlog():
    fp = "heart.txt"
    raw_data = pd.read_csv(fp, delim_whitespace=True, header=None)
    x = np.array(raw_data.iloc[:,0:13])
    y = np.array(raw_data.iloc[:, 13])

    return x, y

if __name__ == '__main__':
    data, target = load_statlog()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
