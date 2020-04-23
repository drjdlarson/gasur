# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:35:41 2020

@author: ryan4
"""
from enum import Enum, auto, unique


@unique
class GuidanceType(Enum):
    ILQR = auto()
    CILQR = auto()
    ELQR = auto()
    NONE = auto()
