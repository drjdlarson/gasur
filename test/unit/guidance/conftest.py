# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:55:41 2020

@author: ryan4
"""
import pytest
import numpy as np

from gasur.estimator import GaussianMixture
from gasur.guidance import GaussianObject


@pytest.fixture(scope="session")
def Q():
    return 10**-3 * np.eye(2)


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def waypoints():
    return waypoint_helper().copy()


def waypoint_helper():
    p1 = np.array([23, 0, 0, 0]).reshape((4, 1))
    p2 = np.array([0, 15, 0, 0]).reshape((4, 1))
    p3 = np.array([-15, 0, 0, 0]).reshape((4, 1))
    p4 = np.array([0, -15, 0, 0]).reshape((4, 1))
    return [p1, p2, p3, p4]


@pytest.fixture(scope="session")
def dyn_funcs():
    def f1(x, u, **kwargs):
        dt = kwargs['dt']
        return x[0] + dt * x[2]

    def f2(x, u, **kwargs):
        dt = kwargs['dt']
        return x[1] + dt * x[3]

    def f3(x, u, **kwargs):
        return x[2] + u[0]

    def f4(x, u, **kwargs):
        return x[3] + u[1]

    return [f1, f2, f3, f4]


@pytest.fixture(scope="session")
def inv_dyn_funcs():
    def f1(x, u, **kwargs):
        dt = kwargs['dt']
        return x[0] - dt * (x[2] - u[0])

    def f2(x, u, **kwargs):
        dt = kwargs['dt']
        return x[1] - dt * (x[3] - u[1])

    def f3(x, u, **kwargs):
        return x[2] - u[0]

    def f4(x, u, **kwargs):
        return x[3] - u[1]

    return [f1, f2, f3, f4]


@pytest.fixture(scope="function")
def cur_gaussians():
    cur_states = np.array([[-6.61516208446766, 7.31118852531110,
                            -0.872274587078559],
                           [0.120718435587983, 0.822267374847013,
                            8.03032533284825],
                           [-3.70905023905814, 0.109099775533068,
                            0.869292134509202],
                           [-0.673511914125811, 0.0704886302223785,
                            1.37195762201929]])
    c1 = np.array([[0.159153553514662, -0.013113139329301,
                    0.067998235869292, -0.002599022295748],
                   [-0.013113139329301, 1.091418243231293,
                    -0.004852173475548, 0.242777767799711],
                   [0.067998235869292, -0.004852173475548,
                    0.052482674913465, -0.001137669337119],
                   [-0.002599022295748, 0.242777767799711,
                    -0.001137669337119, 0.182621583380687]])
    c2 = np.array([[0.168615126193114, 0.000427327257979604,
                    0.0813173846497561, 3.60271394071290e-05],
                   [0.000427327257979604, 1.63427398127019,
                    0.000214726953674745, 0.131506716663899],
                   [0.0813173846497561, 0.000214726953674745,
                    0.168683528526021, 1.81032165630712e-05],
                   [3.60271394071290e-05, 0.131506716663899,
                    1.81032165630712e-05, 0.240588802794357]])
    c3 = np.array([[0.172523281175800, -0.0880334775266906,
                    0.0824376519837369, -0.0127439165441394],
                   [-0.0880334775266907, 1.17602498884724,
                    -0.0323162663735516, 0.221483755921902],
                   [0.0824376519837369, -0.0323162663735516,
                    0.0792669538504277, -0.00591837975207081],
                   [-0.0127439165441394, 0.221483755921902,
                    -0.00591837975207081, 0.206656002000548]])
    cov = [c1, c2, c3]
    cur_gaussians = []
    weights = [0.326646270305158, 0.364094073944246, 0.309259655750595]
    ctr1 = np.array([0.798419210129032, 0.803706982685878])
    ctr2 = np.array([0., 0.])
    ctr3 = np.array([-0.700669070490184, 1.52231152743820])
    ctr = [ctr1, ctr2, ctr3]
    for ii in range(0, 3):
        gm = GaussianObject()
        gm.means = np.zeros((3, 4))
        gm.means[:, :] = cur_states[:, ii].reshape((1, 4))
        gm.ctrl_inputs = np.zeros((3, 2))
        gm.ctrl_inputs[:, :] = ctr[ii]
        gm.covariance = cov[ii]
        gm.weight = weights[ii]
        cur_gaussians.append(gm)
    return cur_gaussians
