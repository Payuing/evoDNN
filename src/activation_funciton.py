import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    y = np.empty_like(x)
    y[:] = x[:]
    y[y <= 0] = 0


def lrelu(x, alpha):
    y = np.empty_like(x)
    y[:] = x[:]
    y[y <= 0] = alpha * x[x <= 0]
    return y
