import numpy as np

def cross_entropy(predictions, targets):
    ce = -np.sum(targets * np.log(predictions))
    return ce

def sigmoid_derivative(layer):
    return layer * (1 - layer)

def sigmoid(x):
    y = np.exp(-x)
    return np.divide(1.0, 1 + y)


def softmax(x):
    y = np.exp(x)
    return y / y.sum()
