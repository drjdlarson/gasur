# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:57:21 2020

@author: ryan4
"""
from abc import ABC, abstractmethod
from ..enumerations import GuidanceType


class Guidance(ABC):

    def __init__(self, guidancetype=GuidanceType.NONE, horizon_steps=0):
        self.type = guidancetype
        self.time_horizon_steps = horizon_steps

    @abstractmethod
    def reinitialize_trajectories(self):
        pass

    @abstractmethod
    def update(self):
        pass
