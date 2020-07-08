class GaussianMixture:
    def __init__(self, **kwargs):
        self.means = kwargs.get('means', [])
        self.covariances = kwargs.get('covariances', [])
        self.weights = kwargs.get('weights', [])
