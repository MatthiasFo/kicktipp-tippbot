import numpy as np


def logistic_fun(x, a, b, c, d):
    return non_neg_logistic(x, a, b, c) + d


def non_neg_logistic(x, alpha, beta, gamma):
    return gamma / (1.0 + np.exp(np.dot(beta, np.add(x, alpha))))


def straight_line(x, a, b):
    return np.add(np.multiply(x, a), b)


def quadratic_fun(x, a, b, c):
    return a * x ** 2 + b * x + c