# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 07:40:41 2020

@author: ryan4
"""
import pytest
import unittest
import numpy as np

from gasur.guidance.base import LQR
from gasur.exceptions import IncorrectNumberOfTargets


# hanldes initialization common to all test cases
class test_GuidanceLQR(unittest.TestCase):
    def setUp(self):
        target1 = np.array([[15], [0], [0], [0]])
        target2 = np.array([[0], [15], [0], [0]])
        target3 = np.array([[-15], [0], [0], [0]])
        target4 = np.array([[0], [-15], [0], [0]])
        self.init_targets = np.hstack((target1, target2, target3, target4))

        cov1 = np.array([[30, 0, 0, 0], [0, 15, 0, 0], [0, 0, 100, 0],
                         [0, 0, 0, 100]])
        cov2 = np.array([[15, 0, 0, 0], [0, 30, 0, 0], [0, 0, 100, 0],
                         [0, 0, 0, 100]])
        cov3 = np.array([[30, 0, 0, 0], [0, 15, 0, 0], [0, 0, 100, 0],
                         [0, 0, 0, 100]])
        cov4 = np.array([[15, 0, 0, 0], [0, 30, 0, 0], [0, 0, 100, 0],
                         [0, 0, 0, 100]])
        self.init_covariances = np.stack((cov1, cov2, cov3, cov4), axis=2)
        self.init_weights = np.array([1, 1, 1, 1])
        self.test_class = LQR(safety_factor=2, target_states=self.init_targets,
                              target_covariances=self.init_covariances,
                              target_weights=self.init_weights)
        self.obj_states = np.array([[-6.795418633599303],
                                    [4.113836424252542],
                                    [-0.415244189894567],
                                    [-0.033063840677620]])
        self.obj_weights = np.array([1])
        self.obj_covariances = np.array([[0.176794403473178,
                                          -0.037461642284415,
                                          0.077810388970633,
                                          -0.003158391405413],
                                        [-0.037461642284415, 1.532926224143814,
                                         -0.018808803324518,
                                         0.109070781732398],
                                        [-0.018808803324518, 0.109070781732398,
                                         0.174592084740625,
                                         -0.001586707394357],
                                        [-0.003158391405413, 0.109070781732398,
                                         -0.001586707394357,
                                         0.239939500510964]])


# test the updateTargets function
class test_updateTargets(test_GuidanceLQR):
    def test_tooFewTargets(self):
        with pytest.raises(IncorrectNumberOfTargets):
            self.test_class.update_targets(self.init_targets[:, 0:3],
                                           self.init_covariances,
                                           self.init_weights)

    def test_tooFewCovariances(self):
        with pytest.raises(IncorrectNumberOfTargets):
            self.test_class.update_targets(self.init_targets,
                                           self.init_covariances[:, :, 0:1],
                                           self.init_weights)


class test_targetCenter(test_GuidanceLQR):
    def test_targetCenter(self):
        res = self.test_class.target_center()
        exp = np.array([[0], [0], [0], [0]])
        self.assertEqual(res, exp)


class test_costFunction(test_GuidanceLQR):
    def test_costFunction(self):
        res = self.cost_function(self.obj_weights, self.obj_states,
                                 self.obj_covariances)
        exp = 1.027689163725033
        self.assertTrue(np.abs(res - exp) <= 1*10**-7)
