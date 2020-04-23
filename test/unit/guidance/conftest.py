# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:55:41 2020

@author: ryan4
"""
import pytest
import numpy as np


@pytest.fixture(scope="session")
def Q():
    return 10**-3 * np.eye(2)


@pytest.fixture(scope="session")
def R():
    return np.array([0.1])
