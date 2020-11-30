import numpy as np
from numpy.linalg import det

from gncpy.math import gamma_fnc


class GaussianMixture:
    """ Gaussian Mixture object

    Attributes:
        means (list): List of Gaussian means, each is a N x 1 numpy array
        covariances (list): List of Gaussian covariances, each is a N x N
            numpy array
        weights (list): List of Gaussian weights, no automatic normalization
    """

    def __init__(self, **kwargs):
        self.means = kwargs.get('means', [])
        self.covariances = kwargs.get('covariances', [])
        self.weights = kwargs.get('weights', [])


class StudentsTMixture:
    """ Students T mixture object.

    Attributes:
        means (list): List of students T means, each is a N x 1 numpy array
        weights (list): List of students T weights
        scalings (list): List of scaling matrices, each is a numpy array
        dof (float): Degrees of freedom for the Students T distribution
    """

    def __init__(self, **kwargs):
        self.means = kwargs.get('means', [])
        self.scalings = kwargs.get('scalings', [])
        self.weights = kwargs.get('weights', [])
        self.dof = kwargs.get('dof', 3)

    @property
    def covariances(self):
        """ List of covariance matrices, each element is a numpy array
        """
        if self.dof <= 2:
            msg = 'Degrees of freedom is {} and must be > 2'
            raise RuntimeError(msg.format(self.dof))
        scale = self.dof / (self.dof - 2)
        return [scale * x for x in self.scalings]

    @classmethod
    def pdf(x, mu, sig, v):
        d = x.size()
        del2 = (x - mu).T @ sig @ (x - mu)
        inv_det = 1 / np.sqrt(det(sig))
        gam_rat = gamma_fnc((v + 2) / 2) / gamma_fnc(v / 2)
        return gam_rat / (v * np.pi)**(d/2) * inv_det \
            * (1 + del2 / v)**(-(v + 2) / 2)


class ParticleDistribution:
    """ Particle distribution object

    Helper class for managing arbitrary distributions of particles.

    Attributes:
        particles (list): List of 2d numpy arrays, one per particle
    """

    def __init__(self, **kwargs):
        self.particles = kwargs.get('particles', [])

    @property
    def mean(self):
        """ Mean of the particles
        """
        if self.num_particles == 0:
            mean = np.array([[]])
        else:
            x_dim = self.particles[0].size
            mean = np.mean(np.hstack(self.particles), axis=1)
            mean = mean.reshape((x_dim, 1))
        return mean

    @property
    def covariance(self):
        """Covariance of the particles

        """
        if self.num_particles == 0:
            cov = np.array([[]])
        else:
            x_dim = self.particles[0].size
            cov = np.cov(np.hstack(self.particles)).reshape((x_dim, x_dim))
        return cov

    @property
    def num_particles(self):
        """ Number of particles
        """
        return len(self.particles)
