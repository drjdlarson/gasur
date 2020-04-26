# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:35:30 2020

@author: ryan4
"""
import pytest
import numpy as np


@pytest.fixture(scope="session")
def func_list():
    def f1(x, u):
        return x[0] * np.sin(x[1]) + 3*x[2]*x[1] + u[0]

    def f2(x, u):
        return x[0]**2 + 3*x[2]*x[1] + u[0] * u[1]

    def f3(x, u):
        return x[2] * np.cos(x[0]) + x[1]**2 + np.sin(u[0])

    return [f1, f2, f3]


@pytest.fixture(scope="session")
def x_point():
    return np.array([[3],
                    [2.7],
                    [-6.25]])


@pytest.fixture(scope="session")
def u_point():
    return np.array([[-0.5],
                     [2]])
