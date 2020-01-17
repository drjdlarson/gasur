# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:20:30 2020

@author: ryan4
"""
from .base import Guidance
from ..enumerations import GuidanceType


class ExtendedLQR(Guidance):
    def __init__(self):
        super().__init__(guidancetype=GuidanceType.ELQR)

    def reinitialize_trajectories(self):
        pass
