""" Defines utility functions for plotting routines.
"""
import numpy as np
from numpy.linalg import eigh


def calc_error_ellipse(cov, n_sig):
    # get and sort eigne values
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # find rotation angle from positive x-axis, and width/height
    angle = 180 / np.pi * np.arctan2(*vecs[:, 0][::-1])
    width, height = 2 * n_sig * np.sqrt(vals)

    return width, height, angle
