# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:24:30 2020

@author: ryan4
"""
import pytest
import numpy as np
import numpy.testing as test
from copy import deepcopy

from gasur.utilities.distributions import GaussianMixture
import gasur.guidance as guide


class TestDensityBased:
    def test_constructor(self):
        exp_safety_factor = 2
        exp_y_ref = 0.8
        densityBased = guide.DensityBased(safety_factor=exp_safety_factor,
                                          y_ref=exp_y_ref)
        test.assert_allclose(densityBased.safety_factor, exp_safety_factor)
        test.assert_allclose(densityBased.y_ref, exp_y_ref)

        densityBased = guide.DensityBased()
        assert len(densityBased.targets.means) == 0

    def test_target_center(self, wayareas):
        densityBased = guide.DensityBased(wayareas=wayareas)
        exp_center = np.array([2, 0, 0, 0]).reshape((4, 1))

        test.assert_allclose(densityBased.target_center(), exp_center)

    def test_convert_waypoints(self, waypoints):
        densityBased = guide.DensityBased()

        c1 = np.zeros((4, 4))
        c1[0, 0] = 38.
        c1[1, 1] = 15.
        c2 = 4 * np.zeros((4, 4))
        c2[0, 0] = 23.966312915412313
        c2[0, 1] = 0.394911136156014
        c2[1, 0] = 0.394911136156014
        c2[1, 1] = 29.908409144106464
        c3 = c1.copy()
        c4 = c2.copy()
        c4[0, 1] *= -1
        c4[1, 0] *= -1
        exp_covs = [c1, c2, c3, c4]

        assert len(densityBased.targets.means) == 0
        assert len(densityBased.targets.covariances) == 0
        wayareas = densityBased.convert_waypoints(waypoints)
        assert len(wayareas.means) == len(waypoints)
        assert len(densityBased.targets.means) == 0
        assert len(densityBased.targets.covariances) == 0

        for ii, cov in enumerate(wayareas.covariances):
            test.assert_allclose(cov, exp_covs[ii],
                                 err_msg='iteration {}'.format(ii))

    def test_update_targets(self, waypoints, wayareas):
        densityBased = guide.DensityBased()
        densityBased.update_targets(waypoints)
        for ii, mean in enumerate(densityBased.targets.means):
            test.assert_allclose(mean, waypoints[ii],
                                 err_msg='iteration {}'.format(ii))

    def test_density_based_cost(self, wayareas):
        for cov in wayareas.covariances:
            cov[2, 2] = 100.
            cov[3, 3] = 100.
        densityBased = guide.DensityBased(safety_factor=2, wayareas=wayareas)
        obj_states = np.array([[-6.61506208462059, 7.31118852531110,
                                -0.872274587078559],
                               [0.120718435588028, 0.822267374847013,
                                8.03032533284825],
                               [-3.70905023932142, 0.109099775533068,
                                0.869292134497459],
                               [-0.673511914128590, 0.0704886302223785,
                                1.37195762192259]])
        obj_weights = [0.326646270304846, 0.364094073944415, 0.309259655750739]
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
        obj_covariances = [c1, c2, c3]

        exp_cost = 2.705543539958144
        cost = densityBased.density_based_cost(obj_states, obj_weights,
                                               obj_covariances)
        test.assert_approx_equal(cost, exp_cost)


@pytest.mark.incremental
class TestELQRGaussian:
    def test_constructor(self, dyn_funcs, inv_dyn_funcs, wayareas, Q):
        Z_mat4 = np.zeros((4, 4))
        Z_mat24 = np.zeros((2, 4))
        Z_vec4 = np.zeros((4, 1))
        Z_vec2 = np.zeros((2, 1))

        gm1 = guide.GaussianObject()
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

        elqrGaussian = guide.ELQRGaussian(cur_gaussians=[gm1], horizon_len=3)
        assert len(elqrGaussian.gaussians) == 1
        assert len(elqrGaussian.targets.means) == 0

        elqrGaussian = guide.ELQRGaussian(cur_gaussians=[gm1],
                                          wayareas=wayareas, Q=Q,
                                          horizon_len=3)
        assert len(elqrGaussian.gaussians) == 1
        assert len(elqrGaussian.targets.means) == len(wayareas.means)
        test.assert_allclose(elqrGaussian.state_penalty, Q)

        elqrGaussian = guide.ELQRGaussian(horizon_len=3)
        assert len(elqrGaussian.gaussians) == 0
        assert elqrGaussian.state_penalty.size == 0
        test.assert_approx_equal(elqrGaussian.safety_factor, 1.)
        assert len(elqrGaussian.targets.means) == 0

        with pytest.raises(RuntimeError, match=r"Horizon .* ELQR"):
            elqrGaussian = guide.ELQRGaussian(horizon_len=np.inf)

    def test_initialize(self):
        meas = GaussianMixture()
        meas.means = [np.array([[1], [2], [3]]), np.array([[4], [5], [6]])]
        meas.covariances = [2*np.eye(3), 2*np.eye(3)]
        meas.weights = [1, 1]
        dyn_lst = [[], []]  # list of lists of functions
        inv_dyn_lst = [[], []]  # list of lists of functions
        n_inputs = [1, 1]
        elqrGaussian = guide.ELQRGaussian()

        elqrGaussian.initialize(meas, dyn_lst, inv_dyn_lst, n_inputs)
        assert len(elqrGaussian.gaussians) == len(meas.means)
        for ii, gg in enumerate(elqrGaussian.gaussians):
            test.assert_allclose(gg.means[[0], :].T, meas.means[ii],
                                 err_msg='Iteration number: {}'.format(ii))

        obj = guide.GaussianObject()
        obj.means = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) + 0.0001
        obj.covariance = np.eye(3) + 0.01
        obj.weight = 1
        obj.dyn_functions = []
        obj.inv_dyn_functions = []
        obj.ctrl_inputs = np.array([[1], [1], [1]])
        obj.feedforward_lst = [np.zeros((1, 1)), np.zeros((1, 1)),
                               np.zeros((1, 1))]
        obj.feedback_lst = [np.zeros((1, 3)), np.zeros((1, 3)),
                            np.zeros((1, 3))]
        obj.cost_to_come_mat = [np.zeros((3, 3)), np.zeros((3, 3)),
                                np.zeros((3, 3))]
        obj.cost_to_come_vec = [np.zeros((3, 1)), np.zeros((3, 1)),
                                np.zeros((3, 1))]
        obj.cost_to_go_mat = [np.zeros((3, 3)), np.zeros((3, 3)),
                              np.zeros((3, 3))]
        obj.cost_to_go_vec = [np.zeros((3, 1)), np.zeros((3, 1)),
                              np.zeros((3, 1))]
        obj.ctrl_nom = np.array([[0]])

        elqrGaussian = guide.ELQRGaussian(cur_gaussians=[obj])

        elqrGaussian.initialize(meas, dyn_lst, inv_dyn_lst, n_inputs)
        assert len(elqrGaussian.gaussians) == len(meas.means)

        obj2 = deepcopy(obj)
        obj2.means = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]]) + 0.000001
        elqrGaussian = guide.ELQRGaussian(cur_gaussians=[obj, obj2])

        elqrGaussian.initialize(meas, dyn_lst, inv_dyn_lst, n_inputs)

        for ii, gg in enumerate(elqrGaussian.gaussians):
            test.assert_allclose(gg.means[[0], :].T, meas.means[ii],
                                 err_msg='Iteration number: {}'.format(ii))

    def test_quadratize_non_quad_state(self, wayareas, cur_gaussians):
        cur_states = np.zeros((4, 3))
        for ii, gg in enumerate(cur_gaussians):
            cur_states[:, ii] = gg.means[0, :]
        for cov in wayareas.covariances:
            cov[2, 2] = 100.
            cov[3, 3] = 100.
        wayareas.weights = [0.5, 0.5, 0.5, 0.5]
        obj_num = 0
        elqrGaussian = guide.ELQRGaussian(cur_gaussians=cur_gaussians,
                                          wayareas=wayareas, horizon_len=3)

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

        test.assert_allclose(Q, exp_Q, atol=3e-8)
        test.assert_allclose(q, exp_q, atol=3e-8)

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
            gm = guide.GaussianObject()
            gm.means = cur_states[:, ii].reshape((1, 4))
            gm.ctrl_inputs = ctr[ii]
            gm.covariance = cov[ii]
            gm.weight = weights[ii]
            cur_gaussians.append(gm)
        for cov in wayareas.covariances:
            cov[2, 2] = 100.
            cov[3, 3] = 100.
        wayareas.weights = [0.5, 0.5, 0.5, 0.5]
        elqrGaussian = guide.ELQRGaussian(Q=np.eye(4), R=np.eye(2),
                                          cur_gaussians=cur_gaussians,
                                          wayareas=wayareas, horizon_len=3)
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

    def test_find_nearest_target(self, wayareas, waypoints):
        elqrGaussian = guide.ELQRGaussian(Q=np.eye(4), R=np.eye(2),
                                          wayareas=wayareas)
        state = waypoints[0] + 2

        exp_goal = waypoints[0]

        goal = elqrGaussian.find_nearest_target(state)

        test.assert_allclose(goal, exp_goal)

    def test_final_cost_function(self, wayareas, waypoints):
        elqrGaussian = guide.ELQRGaussian(Q=np.eye(4), R=np.eye(2),
                                          wayareas=wayareas)
        states = np.zeros((2, 4))  # num objects by num states
        states[0, :] = waypoints[0].squeeze() + 2
        states[1, :] = waypoints[1].squeeze() + 1

        exp_cost = 20

        cost = elqrGaussian.final_cost_function(states)
        test.assert_approx_equal(cost, exp_cost)

    @pytest.mark.slow
    def test_iterate(self, cur_gaussians, wayareas, dyn_funcs, inv_dyn_funcs):
        meas = GaussianMixture()
        meas.means = [np.array([[1], [2], [3], [4]]),
                      np.array([[4], [5], [6], [7]])]
        meas.covariances = [2*np.eye(4), 2*np.eye(4)]
        meas.weights = [0.3, 0.3, 0.3]
        dyn_lst = [dyn_funcs, dyn_funcs, dyn_funcs]
        inv_dyn_lst = [inv_dyn_funcs, inv_dyn_funcs, inv_dyn_funcs]
        n_inputs = [2, 2, 2]
        for cov in wayareas.covariances:
            cov[2, 2] = 100.
            cov[3, 3] = 100.
        wayareas.weights = [0.5, 0.5, 0.5, 0.5]
        elqrGaussian = guide.ELQRGaussian(Q=np.eye(4), R=np.eye(2),
                                          cur_gaussians=cur_gaussians,
                                          wayareas=wayareas, horizon_len=3)

        elqrGaussian.iterate(meas, est_dyn_lst=dyn_lst,
                             est_inv_dyn_lst=inv_dyn_lst,
                             n_inputs_lst=n_inputs, dt=1)
