# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:24:30 2020

@author: ryan4
"""
import pytest
import numpy as np

from gasur.guidance.base import BaseLQR


@pytest.mark.incremental
class TestBaseLQR:
    def test_constructor(self, Q, R):
        self.baseLQR = BaseLQR(Q=Q, R=R)
        np.testing.assert_allclose(self.baseLQR.state_penalty, Q)
        np.testing.assert_allclose(self.baseLQR.ctrl_penalty, R)

    def test_iterate(self, Q, R):
        self.baseLQR = BaseLQR(Q=Q, R=R)
        F = np.array([[1, 0.5],
                      [0, 1]])
        G = np.array([[1],
                      [1]])
        expected_K = np.array([[0.0843830939647654, 0.245757798310976]])
        K = self.baseLQR.iterate(F=F, G=G)
        np.testing.assert_allclose(K, expected_K)
