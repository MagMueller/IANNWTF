import numpy as np
import math


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


def sigmoidprime(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds
