# Leander Berg Thorkildsen
# 17. September 2019
# Assignment 2 - Generate data-sets based on logic gates


import numpy as np


def generate_data(operator="AND", n_sets=100, noise=0.0):
    x = np.random.randint(0, 2, size=(n_sets, 2))
    y = None  # Logic output is based on operator

    if operator == "AND":
        y = np.logical_and(x[:, 0:1], x[:, 1:2])
    if operator == "OR":
        y = np.logical_or(x[:, 0:1], x[:, 1:2])
    if operator == "XOR":
        y = np.logical_xor(x[:, 0:1], x[:, 1:2])
        fraction = np.random.rand(n_sets, 1) < noise
        y = np.logical_xor(y, fraction)

    return x, y


def generate_test():
    X = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ])

    y = np.array([1, 0, 0, 0])

    return X, y
