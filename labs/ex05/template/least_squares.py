# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
    
    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum
    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def least_squares(y, tx):
    """calculate the least squares."""
    xtx = tx.T@tx
    xty = tx.T@y
    w = np.linalg.solve(xtx, xty)
    mse = compute_mse(y, tx, w)
    return w, mse
