"""Standardized implementations of common distribution objects.

These distributions are specific to RFS swarms and are often mixtures.
Other distributions can be found in GNCPy.
"""
import numpy as np
from numpy.linalg import det
import numpy.random as rnd
import scipy.stats as stats

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

    def __init__(self, means=None, covariances=None, weights=None):
        if means is None:
            means = []
        if covariances is None:
            covariances = []
        if weights is None:
            weights = []
        self.means = means
        self.covariances = covariances
        self.weights = weights

    def sample(self, rng=None):
        """Draw a sample from the current mixture model.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        numpy array
            randomly sampled numpy array of the same shape as the mean.
        """
        if rng is None:
            rng = rnd.default_rng()
        mix_ind = rng.choice(np.arange(len(self.means), dtype=int),
                             p=self.weights)
        x = rng.multivariate_normal(self.means[mix_ind].flatten(),
                                    self.covariances[mix_ind])
        return x.reshape(self.means[mix_ind].shape)

    def pdf(self, x):
        """Multi-variate probability density function for this mixture.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_normal
        flat_x = x.flatten()
        p = 0
        for m, s, w in zip(self.means, self.covariances, self.weights):
            p += w * rv.pdf(flat_x, mean=m.flatten(), cov=s)

        return p


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

    def __init__(self, means=None, scalings=None, weights=None, dof=3):
        if means is None:
            means = []
        if scalings is None:
            scalings = []
        if weights is None:
            weights = []
        self.means = means
        self.scalings = scalings
        self.weights = weights
        self.dof = dof

    @property
    def covariances(self):
        """List of covariance matrices, each element is a numpy array."""
        if self.dof <= 2:
            msg = 'Degrees of freedom is {} and must be > 2'
            raise RuntimeError(msg.format(self.dof))
        scale = self.dof / (self.dof - 2)
        return [scale * x for x in self.scalings]

    def pdf(self, x):
        """Multi-variate probability density function for this mixture.

        Parameters
        ----------
        x : N x 1 numpy array
            Value to evaluate the pdf at.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_t
        flat_x = x.flatten()
        p = 0
        for m, s, w in zip(self.means, self.scalings, self.weights):
            p += w * rv.pdf(flat_x, loc=m.flatten(), shape=s, df=self.dof)

        return p

    def sample(self, rng=None):
        """Multi-variate probability density function for this mixture.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        if rng is None:
            rng = rnd.default_rng()

        rv = stats.multivariate_t
        rv.random_state = rng
        mix_ind = rng.choice(np.arange(len(self.means), dtype=int),
                             p=self.weights)
        if isinstance(self.dof, list):
            df = self.dof[mix_ind]
        else:
            df = self.dof
        x = rv.rvs(loc=self.means[mix_ind].flatten(),
                   shape=self.scalings[mix_ind], df=df)

        return x.reshape(self.means[mix_ind].shape)
