# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((degree+1, x.shape[0]))
    for i in range(1, degree+1):
        poly[i,:] = poly[i-1,:]*x
    return poly.T
