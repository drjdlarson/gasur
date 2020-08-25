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
