""" Defines utility functions for plotting routines.
"""
import numpy as np
from numpy.linalg import eigh


def calc_error_ellipse(cov, n_sig):
    """ Calculates parameters for an error ellipse.
    
    This calucates the error ellipse for a given sigma
    number according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`.
    
    Args:
        cov (2 x 2 numpy array): covariance matrix.
        n_sig (float): Sigma number, must be positive.
    
    Returns:
        tuple containing

                - width (float): The width of the ellipse
                - height (float): The height of the ellipse
                - angle (float): The rotation angle in degrees
                of the semi-major axis. Measured up from the
                positive x-axis.
    """
    # get and sort eigne values
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # find rotation angle from positive x-axis, and width/height
    angle = 180 / np.pi * np.arctan2(*vecs[:, 0][::-1])
    width, height = 2 * n_sig * np.sqrt(vals)

    return width, height, angle
