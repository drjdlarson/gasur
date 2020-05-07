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

#    def test_initialize(self):
#        assert 0, 'implement'

    def test_quadratize_non_quad_state(self, wayareas):
        cur_states = np.array([[-6.61516208455517, 7.31118852531110,
                                -0.872274587078559],
                               [0.120718435589312, 0.822267374847013,
                                8.03032533284825],
                               [-3.70905023923744, 0.109099775533068,
                                0.869292134497581],
                               [-0.673511914127673, 0.0704886302223785,
                                1.37195762193361]])
        obj_num = 0
        c1 = np.array([[0.159153553514662, -0.0131131393293015,
                        0.0679982358692922, -0.00259902229574857],
                       [-0.0131131393293015, 1.09141824323191,
                        -0.00485217347554817, 0.242777767799848],
                       [0.0679982358692923, -0.00485217347554816,
                        0.0524826749134647, -0.00113766933711917],
                       [-0.00259902229574857, 0.242777767799848,
                        -0.00113766933711917, 0.182621583380717]])
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
        weights = [0.326646270304980, 0.364094073944343, 0.309259655750677]
        ctr1 = np.array([0.798419210129032, 0.803706982685878])
        ctr2 = np.array([0., 0.])
        ctr3 = np.array([-0.700669070490184, 1.52231152743820])
        ctr = [ctr1, ctr2, ctr3]
        for ii in range(0, 3):
            gm = GaussianObject()
            gm.means = cur_states[:, ii].reshape((1, 4))
            gm.ctrl_inputs = ctr[ii]
            gm.covariance = cov[ii]
            gm.weight = weights[ii]
            cur_gaussians.append(gm)
            wayareas.weights = [0.5, 0.5, 0.5, 0.5]
        elqrGaussian = guide.ELQRGaussian(cur_gaussians=cur_gaussians,
                                          wayareas=wayareas)

        exp_Q = np.array([[6.08618287e-08, -7.62808745e-06, -1.31408886e-07,
                           -5.24016660e-10],
                          [-7.62808745e-06,  9.56062598e-04,  1.64700683e-05,
                           6.56773710e-08],
                          [-1.31408886e-07,  1.64700683e-05,  2.83729487e-07,
                           1.13142255e-09],
                          [-5.24016660e-10,  6.56773710e-08,  1.13142255e-09,
                           4.51175171e-12]])
        exp_q = np.array([[7.570267e-03],
                          [1.793011e-05],
                          [1.558761e-04],
                          [2.919833e-05]])

        Q, q = elqrGaussian.quadratize_non_quad_state(all_states=cur_states,
                                                      obj_num=obj_num)

        test.assert_allclose(Q, exp_Q)
        test.assert_allclose(q, exp_q, atol=1e-8)

    def test_quadratize_cost(self, wayareas):
        cur_states = np.array([[-6.61516208455517, 7.31118852531110,
                                -0.872274587078559],
                               [0.120718435589312, 0.822267374847013,
                                8.03032533284825],
                               [-3.70905023923744, 0.109099775533068,
                                0.869292134497581],
                               [-0.673511914127673, 0.0704886302223785,
                                1.37195762193361]])
        obj_num = 0
        c1 = np.array([[0.159153553514662, -0.0131131393293015,
                        0.0679982358692922, -0.00259902229574857],
                       [-0.0131131393293015, 1.09141824323191,
                        -0.00485217347554817, 0.242777767799848],
                       [0.0679982358692923, -0.00485217347554816,
                        0.0524826749134647, -0.00113766933711917],
                       [-0.00259902229574857, 0.242777767799848,
                        -0.00113766933711917, 0.182621583380717]])
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
        weights = [0.326646270304980, 0.364094073944343, 0.309259655750677]
        ctr1 = np.array([0.798419210129032, 0.803706982685878])
        ctr2 = np.array([0., 0.])
        ctr3 = np.array([-0.700669070490184, 1.52231152743820])
        ctr = [ctr1, ctr2, ctr3]
        for ii in range(0, 3):
            gm = GaussianObject()
            gm.means = cur_states[:, ii].reshape((1, 4))
            gm.ctrl_inputs = ctr[ii]
            gm.covariance = cov[ii]
            gm.weight = weights[ii]
            cur_gaussians.append(gm)
            wayareas.weights = [0.5, 0.5, 0.5, 0.5]
        elqrGaussian = guide.ELQRGaussian(Q=np.eye(4), R=np.eye(2),
                                          cur_gaussians=cur_gaussians,
                                          wayareas=wayareas)
        x_start = np.zeros((4, 1))
        u_nom = np.zeros((2, 1))

        exp_Q = np.array([[6.08618287e-08, -7.62808745e-06, -1.31408886e-07,
                           -5.24016660e-10],
                          [-7.62808745e-06,  9.56062598e-04,  1.64700683e-05,
                           6.56773710e-08],
                          [-1.31408886e-07,  1.64700683e-05,  2.83729487e-07,
                           1.13142255e-09],
                          [-5.24016660e-10,  6.56773710e-08,  1.13142255e-09,
                           4.51175171e-12]])
        exp_q = np.array([[7.570267e-03],
                          [1.793011e-05],
                          [1.558761e-04],
                          [2.919833e-05]])

        P, Q, R, q, r = elqrGaussian.quadratize_cost(x_start, u_nom, 1,
                                                     all_states=cur_states,
                                                     obj_num=obj_num)

        test.assert_allclose(Q, exp_Q)
        test.assert_allclose(q, exp_q, atol=1e-8)

    def test_forward_pass(self):
        assert 0, 'implement'

    def test_backward_pass(self):
        assert 0, 'implement'

    def test_iterate(self):
        assert 0, 'implement'
