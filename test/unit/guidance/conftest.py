# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:55:41 2020

@author: ryan4
"""
import pytest
import numpy as np

from gasur.estimator import GaussianMixture


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


@pytest.fixture(scope="session")
def wayareas():
    means = waypoint_helper().copy()

    c1 = np.zeros((4, 4))
    c1[0, 0] = 38
    c1[1, 1] = 15
    c2 = 4 * np.zeros((4, 4))
    c2[0, 0] = 23.966312915412313
    c2[0, 1] = 0.394911136156014
    c2[1, 0] = 0.394911136156014
    c2[1, 1] = 29.908409144106464
    c3 = c1.copy()
    c4 = c2.copy()
    c4[0, 1] *= -1
    c4[1, 0] *= -1
    covs = [c1, c2, c3, c4]

    weights = [0.25, 0.25, 0.25, 0.25]

    return GaussianMixture(means, covs, weights)


@pytest.fixture(scope="session")
def waypoints():
    return waypoint_helper().copy()


def waypoint_helper():
    p1 = np.array([23, 0, 0, 0]).reshape((4, 1))
    p2 = np.array([0, 15, 0, 0]).reshape((4, 1))
    p3 = np.array([-15, 0, 0, 0]).reshape((4, 1))
    p4 = np.array([0, -15, 0, 0]).reshape((4, 1))
    return [p1, p2, p3, p4]
