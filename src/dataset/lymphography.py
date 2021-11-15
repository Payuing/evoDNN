"""
 This lymphography domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. (Restricted access)
"""
import numpy as np, pandas as pd

def load_lymphography():
    fp = 'lymphography.csv'
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(raw_data.iloc[:, 0])
    return x, y


if __name__ == '__main__':
    data, target = load_lymphography()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
