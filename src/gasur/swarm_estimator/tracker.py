import numpy as np
import abc

from gncpy.filters import BayesFilter
from gasur.utilities.distributions import GaussianMixture


class RandomFiniteSetBase(metaclass=abc.ABCMeta):
    """ Generic base class for RFS based filters.

    Attributes:
        filter (gncpy.filters.BayesFilter): Filter handling dynamics
    """
    def __init__(self, **kwargs):
        self.filter = BayesFilter()
        self.prob_detection = 1
        self.birth_terms = []
        super.__init__(**kwargs)

    @property
    def prob_miss_detection(self):
        return 1 - self.prob_detection

    @property
    def num_birth_terms(self):
        return len(self.birth_terms)

    @abc.abstractmethod
    def predict(self, **kwargs):
        pass

    @abc.abstractmethod
    def correct(self, **kwargs):
        pass


class GeneralizedLabeledMultiBernoulli(RandomFiniteSetBase):
    """ Delta-Generalized Labeled Multi-Bernoulli filter.

    This is based on :cite:`Vo2013_LabeledRandomFiniteSetsandMultiObjectConjugatePriors`
    and :cite:`Vo2014_LabeledRandomFiniteSetsandtheBayesMultiTargetTrackingFilter`
    """
    def __init__(self, **kwargs):
        self.max_births = 0 # filter.H_bth
        super().__init__(**kwargs)

    def predict(self, **kwargs):
        assert 0

    def correct(self, **kwargs):
        assert 0
