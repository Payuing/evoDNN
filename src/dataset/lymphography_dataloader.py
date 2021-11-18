"""
 This lymphography domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. (Restricted access)
"""
import numpy as np
import pandas as pd
from dataset_peek import data_peek


def load_lymphography():
    fp = 'lymphography.csv'
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(raw_data.iloc[:, 0])
    return x, y


if __name__ == '__main__':
    data_peek("Lymphography", load_lymphography)
