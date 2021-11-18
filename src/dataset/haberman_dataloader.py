"""
Dataset contains cases from study conducted on the survival of patients who had undergone surgery for breast cancer
"""
import numpy as np
import pandas as pd
from dataset_peek import data_peek

def load_haberman():
    fp = 'haberman.csv'
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, :3])
    y = np.array(raw_data.iloc[:, 3])
    return x, y


if __name__ == '__main__':
    data_peek("Haberman", load_haberman)
