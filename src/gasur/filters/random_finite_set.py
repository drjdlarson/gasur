import numpy as np
import abc

from gncpy.filters import BayesFilter


class RandomFiniteSetBase(metaclass=abc.ABCMeta):
    """ Generic base class for RFS based filters.

    Attributes:
        filter (gncpy.gilters.BayesFilter): Filter handling dynamics
    """
    def __init__(self, **kwargs):
        self.filter = BayesFilter(**kwargs)
        super.__init__(**kwargs)

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
        super().__init__(**kwargs)

    def predict(self, **kwargs):
        assert 0

    def correct(self, **kwargs):
        assert 0
