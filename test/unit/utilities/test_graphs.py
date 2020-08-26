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
    def test_bfm_shortest_path1(self):
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

    def test_bfm_shortest_path2(self):
        eps = np.finfo(float).eps
        ncm = np.array([[0., 3.89182030, 3.89182030, 3.47609869,
                         3.47609869e+00, np.inf],
                       [0., 0., 3.89182030, 3.47609869, 3.47609869, eps],
                       [0., 0., 0., 3.47609869, 3.47609869, eps],
                       [0., 0., 0., 0., 3.47609869, eps],
                       [0., 0., 0., 0., 0., eps],
                       [0., 0., 0., 0., 0., 0.]])
        src = 0
        dst = 5

        (cost, path, _) = graphs.bfm_shortest_path(ncm, src, dst)

        exp_cost = 3.47609868983527
        exp_path = [0, 3, 5]

        test.assert_approx_equal(cost, exp_cost)
        test.assert_array_equal(np.array(path), np.array(exp_path))

    def test_k_shortest(self):
        eps = np.finfo(float).eps
        k = 5
        log_cost = np.array([3.89182029811063, 3.89182029811063,
                             3.47609868983527, 3.47609868983527])

        (paths, costs) = graphs.k_shortest(log_cost, k)

        exp_paths = [[], [3], [2], [1], [0]]
        exp_costs = [eps, 3.47609868983527, 3.47609868983527,
                     3.89182029811063, 3.89182029811063]

        assert len(paths) == len(exp_paths)
        for ii in range(0, len(paths)):
            test.assert_array_equal(np.array(paths[ii]),
                                    np.array(exp_paths[ii]))

        test.assert_array_almost_equal(np.array(costs).squeeze(),
                                       np.array(exp_costs))
