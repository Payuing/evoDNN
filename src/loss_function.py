import numpy as np


def rmse(true, predict):
    predict = predict.reshape(np.empty_like(true))
    return np.sqrt(np.mean(np.power(predict - true), 2))

def identity(x):
    return x[:]




class LossFunction:
    def __init__(self):
        pass

