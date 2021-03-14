import numpy as np


def evaluate_kicktipp_432(actual, pred):
    if pred[0] == actual[0] and pred[1] == actual[1]:
        return 4
    elif (pred[0] - pred[1]) == (actual[0] - actual[1]):
        return 3
    elif np.sign(pred[0] - pred[1]) == np.sign(actual[0] - actual[1]):
        return 2
    else:
        return 0


def evaluate_kicktipp_432_vectorized(actual, pred):
    actual_gd = actual[:, 0] - actual[:, 1]
    pred_gd = pred[0] - pred[1]
    points = 2.0 * (np.sign(actual_gd) == np.sign(pred_gd)) + 1.0 * (actual_gd == pred_gd) + \
             1.0 * ((actual[:, 0] == pred[0]) * (actual[:, 1] == pred[1]))
    return points