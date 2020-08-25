# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 19:12:11 2020

@author: ryan4
"""
import pytest
import numpy as np
import numpy.testing as test

import gasur.utilities.graphs as graphs


@pytest.mark.incremental
class TestKShortesst:
    def test_bfm_shortest_path(self):
        eps = np.finfo(float).eps
        ncm = np.array([[0., 3.8918, 3.8918, 3.4761, 3.4761, eps],
                        [0., 0., 3.8918, 3.4761, 3.4761, eps],
                        [0., 0., 0., 3.4761, 3.4761, eps],
                        [0., 0., 0., 0., 3.4761, eps],
                        [0., 0., 0., 0., 0., eps],
                        [0., 0., 0., 0., 0., 0]])
        src = 0
        dst = 5

        (cost, path, pred) = graphs.bfm_shortest_path(ncm, src, dst)

        exp_cost = eps
        exp_path = [0, 5]

        test.assert_approx_equal(cost, exp_cost)
        test.assert_array_equal(np.array(path), np.array(exp_path))
