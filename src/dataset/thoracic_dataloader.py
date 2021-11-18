"""
he data is dedicated to classification problem related to the post-operative life expectancy in the lung cancer patients: class 1 - death within one year after surgery, class 2 - survival.
"""
import numpy as np
import pandas as pd
from dataset_peek import data_peek


def load_thoracic():
    fp = "thoracic.csv"
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 0:16])
    y = np.array(raw_data.iloc[:, 16])
    return x, y


if __name__ == '__main__':
    data_peek("Thoracic", load_thoracic)
