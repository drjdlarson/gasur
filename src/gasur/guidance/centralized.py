# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:22:52 2020

@author: ryan4
"""
from ..estimator import GaussianMixture
from .base import BaseELQR, DensityBased


class ELQRGuassian(BaseELQR, DensityBased):
    def __init__(self, cur_gaussians=[], **kwargs):
        self.cur_guassians = cur_gaussians
        super().__init__(**kwargs)
        
        # ##TODO: implement
        c = self.__class__.__name__
        n = self.iterate.__name__
        msg = '{}.{} not implemented'.format(c, n)
        raise RuntimeError(msg)

    def iterate(self, **kwargs):
        # ##TODO: implement
        msg = '{}.{} not implemented'.format(self.__class__.__name__,
                                             self.iterate.__name__)
        raise RuntimeError(msg)
