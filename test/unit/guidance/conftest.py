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
    def f1(x, u, **kwargs):
        return 2 * x[0] / (1 - x[0]) + u[0]

    def f2(x, u, **kwargs):
        return (1 - 2 * x[1]) / 5 - u[0]**2

    return [f1, f2]


@pytest.fixture(scope="session")
def inv_func_list():
    def f1(x, u, **kwargs):
        return (x[0] - u[0]) / (x[0] + 2 + u[0])

    def f2(x, u, **kwargs):
        return 0.5 * (1 - 5 * x[1] + 5 * u[0]**2)

    return [f1, f2]
