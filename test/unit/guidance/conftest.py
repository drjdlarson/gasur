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
    return np.array([0.1]).reshape((1, 1))


@pytest.fixture(scope="session")
def func_list():
    def f1(x, u):
        out1 = x[0] * np.sin(x[1]) + 3*x[2]*x[1]
        out2 = x[0] * np.cos(x[1]) + x[2]**2
        out3 = x[0]**2 + 3*x[2]*x[1]
        return np.vstack((out1, out2, out3))

    def f2(x, u):
        out1 = x[2] * np.sin(x[1]) + 3*x[2]*x[1]
        out3 = x[1] * np.cos(x[0]) + x[0]**2
        out2 = x[0]**2 + 3*x[2]*x[1]
        return np.vstack((out1, out2, out3))

    def f3(x, u):
        out3 = x[1] * np.sin(x[2]) + 3*x[0]*x[1]
        out2 = x[2] * np.cos(x[0]) + x[1]**2
        out1 = x[0]**2 + 3*x[2]*x[1]
        return np.vstack((out1, out2, out3))

    return [f1, f2, f3]
