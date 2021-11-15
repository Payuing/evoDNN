"""
Dataset contains cases from study conducted on the survival of patients who had undergone surgery for breast cancer
"""
import numpy as np, pandas as pd

def load_haberman():
    fp = 'haberman.csv'
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, :3])
    y = np.array(raw_data.iloc[:, 3])
    return x, y

if __name__ == '__main__':
    data, target = load_haberman()
    print(type(data))
    print(data)
    print(type(target))
    print(target)
