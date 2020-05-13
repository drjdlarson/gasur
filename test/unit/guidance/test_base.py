# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:24:30 2020

@author: ryan4
"""
import numpy as np
import numpy.testing as test


class TestDensityBased:
    def test_constructor(self):
        exp_safety_factor = 2
        exp_y_ref = 0.8
        densityBased = base.DensityBased(safety_factor=exp_safety_factor,
                                         y_ref=exp_y_ref)
        test.assert_allclose(densityBased.safety_factor, exp_safety_factor)
        test.assert_allclose(densityBased.y_ref, exp_y_ref)

        densityBased = base.DensityBased()
        assert len(densityBased.targets.means) == 0


    def test_target_center(self, wayareas):
        densityBased = base.DensityBased(wayareas=wayareas)
        exp_center = np.array([2, 0, 0, 0]).reshape((4, 1))

        test.assert_allclose(densityBased.target_center(), exp_center)

    def test_convert_waypoints(self, waypoints):
        densityBased = base.DensityBased()

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
        densityBased = base.DensityBased()
        densityBased.update_targets(waypoints)
        for ii, mean in enumerate(densityBased.targets.means):
            test.assert_allclose(mean, waypoints[ii],
                                 err_msg='iteration {}'.format(ii))

    def test_density_based_cost(self, wayareas):
        for cov in wayareas.covariances:
            cov[2, 2] = 100.
            cov[3, 3] = 100.
        densityBased = base.DensityBased(safety_factor=2, wayareas=wayareas)
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
