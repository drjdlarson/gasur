"""Standardized implementations of common distribution objects.

These distributions are specific to RFS swarms and are often mixtures.
Other distributions can be found in GNCPy.
"""
import numpy as np
from numpy.linalg import det

from gncpy.math import gamma_fnc


class GaussianMixture:
    """Gaussian Mixture object.

    Attributes
    ----------
    means : list
        List of Gaussian means, each is a N x 1 numpy array
    covariances : list
        List of Gaussian covariances, each is a N x N numpy array
    weights : list
        List of Gaussian weights, no automatic normalization
    """

    def __init__(self, **kwargs):
        self.means = kwargs.get('means', [])
        self.covariances = kwargs.get('covariances', [])
        self.weights = kwargs.get('weights', [])


class StudentsTMixture:
    """Students T mixture object.

    Attributes
    ----------
    means : list
        List of students T means, each is a N x 1 numpy array
    weights : list
        List of students T weights
    scalings : list
        List of scaling matrices, each is a numpy array
    dof : float
        Degrees of freedom for the Students T distribution
    """

    def __init__(self, **kwargs):
        self.means = kwargs.get('means', [])
        self.scalings = kwargs.get('scalings', [])
        self.weights = kwargs.get('weights', [])
        self.dof = kwargs.get('dof', 3)

    @property
    def covariances(self):
        """List of covariance matrices, each element is a numpy array."""
        if self.dof <= 2:
            msg = 'Degrees of freedom is {} and must be > 2'
            raise RuntimeError(msg.format(self.dof))
        scale = self.dof / (self.dof - 2)
        return [scale * x for x in self.scalings]

    @classmethod
    def pdf(x, mu, sig, v):
        """Multi-variate probability density function for the distribution.

        This does not need to be called from a class instance.

        .. todo::
            Allow the STM pdf method to work on mixtures

        Parameters
        ----------
        x : N x 1 numpy array
            Variable to evaluate the PDF at.
        mu : N x 1 numpy array
            Mean of the distribution.
        sig : N x N numpy array
            Scale matrix for the distribution.
        v : int
            Degree of freedom for the distribution, must be >2.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        d = x.size()
        del2 = (x - mu).T @ sig @ (x - mu)
        inv_det = 1 / np.sqrt(det(sig))
        gam_rat = gamma_fnc((v + 2) / 2) / gamma_fnc(v / 2)
        return gam_rat / (v * np.pi)**(d / 2) * inv_det \
            * (1 + del2 / v)**(-(v + 2) / 2)
