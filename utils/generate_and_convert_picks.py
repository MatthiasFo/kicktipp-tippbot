import numpy as np


def get_likely_results():
    picks = np.array(np.meshgrid(range(0, 5), range(0, 5))).T.reshape(-1, 2)
    # remove high draw results
    cleaned_picks = np.append([pick for pick in picks if pick[0] != pick[1]],
                              np.array([[1, 1], [0, 0], [2, 2]]), axis=0)
    return cleaned_picks


def convert_pick_array_to_string(pick):
    return str(int(np.round(pick[0]))) + '-' + str(int(np.round(pick[1])))


def convert_pick_string_to_array(pick):
    return [int(str.split(pick, '-')[0]), int(str.split(pick, '-')[1])]