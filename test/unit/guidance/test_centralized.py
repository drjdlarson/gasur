# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:43:52 2020

@author: ryan4
"""
import pytest
import numpy as np
import numpy.testing as test

from gasur.guidance.base import GaussianObject
import gasur.guidance.centralized as guide


@pytest.mark.incremental
class TestELQRGaussian:
    def test_constructor(self, dyn_funcs, inv_dyn_funcs, wayareas, Q):
        Z_mat4 = np.zeros((4, 4))
        Z_mat24 = np.zeros((2, 4))
        Z_vec4 = np.zeros((4, 1))
        Z_vec2 = np.zeros((2, 1))

        gm1 = GaussianObject()
        gm1.dyn_functions = dyn_funcs
        gm1.inv_dyn_functons = inv_dyn_funcs
        gm1.means = np.vstack((Z_vec4.copy().T, Z_vec4.copy().T,
                               Z_vec4.copy().T))
        gm1.ctrl_inputs = np.vstack((Z_vec2.copy().T, Z_vec2.copy().T,
                                     Z_vec2.copy().T))
        gm1.feedback_lst = [Z_mat24.copy(), Z_mat24.copy(), Z_mat24.copy()]
        gm1.feedforward_lst = [Z_vec2.copy(), Z_vec2.copy(), Z_vec2.copy()]
        gm1.cost_to_come_mat = [Z_mat4.copy(), Z_mat4.copy(), Z_mat4.copy()]
        gm1.cost_to_come_vec = [Z_vec4.copy(), Z_vec4.copy(), Z_vec4.copy()]
        gm1.cost_to_go_mat = gm1.cost_to_come_mat.copy()
        gm1.cost_to_go_vec = gm1.cost_to_come_vec.copy()
        gm1.covariance = Z_mat4.copy()
        gm1.weight = 1

        elqrGaussian = guide.ELQRGaussian(cur_gaussians=[gm1])
        assert len(elqrGaussian.gaussians) == 1
        assert len(elqrGaussian.targets.means) == 0

        elqrGaussian = guide.ELQRGaussian(cur_gaussians=[gm1],
                                          wayareas=wayareas, Q=Q)
        assert len(elqrGaussian.gaussians) == 1
        assert len(elqrGaussian.targets.means) == len(wayareas.means)
        test.assert_allclose(elqrGaussian.state_penalty, Q)

        elqrGaussian = guide.ELQRGaussian()
        assert len(elqrGaussian.gaussians) == 0
        assert elqrGaussian.state_penalty.size == 0
        test.assert_approx_equal(elqrGaussian.safety_factor, 1.)
        assert len(elqrGaussian.targets.means) == 0

    def test_initialize(self):
        assert 0, 'implement'

    def test_quadratize_non_quad_state(self):
        assert 0, 'implement'

    def test_quadratize_cost(self):
        assert 0, 'implement'

    def test_forward_pass(self):
        assert 0, 'implement'

    def test_backward_pass(self):
        assert 0, 'implement'

    def test_iterate(self):
        assert 0, 'implement'
