# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:24:30 2020

@author: ryan4
"""
import pytest
import numpy as np
import numpy.testing as test

import gasur.guidance.base as base


@pytest.mark.incremental
class TestBaseLQR:
    def test_constructor(self, Q, R):
        baseLQR = base.BaseLQR(Q=Q, R=R)
        test.assert_allclose(baseLQR.state_penalty, Q)
        test.assert_allclose(baseLQR.ctrl_penalty, R)

    def test_iterate(self, Q, R):
        baseLQR = base.BaseLQR(Q=Q, R=R)
        F = np.array([[1, 0.5],
                      [0, 1]])
        G = np.array([[1],
                      [1]])
        expected_K = np.array([[0.0843830939647654, 0.245757798310976]])
        K = baseLQR.iterate(F=F, G=G)
        test.assert_allclose(K, expected_K)


@pytest.mark.incremental
class TestBaseELQR:
    def test_constructor(self, Q, R):
        baseELQR = base.BaseELQR(Q=Q, R=R)
        test.assert_allclose(baseELQR.state_penalty, Q)
        test.assert_allclose(baseELQR.ctrl_penalty, R)

        mi = 20
        baseELQR = base.BaseELQR(Q=Q, R=R, max_iters=mi)
        test.assert_allclose(baseELQR.state_penalty, Q)
        test.assert_allclose(baseELQR.ctrl_penalty, R)

        assert baseELQR.max_iters == mi

    def test_initialize(self, Q, R):
        baseELQR = base.BaseELQR(Q=Q, R=R)
        x_start = np.array([[0],
                           [1]])
        n_inputs = 1
        feedback, feedforward, cost_go_mat, cost_go_vec, cost_come_mat, \
            cost_come_vec, x_hat = baseELQR.initialize(x_start=x_start,
                                                       n_inputs=n_inputs)
        exp_feedback = np.zeros((1, 2))
        exp_feedfor = np.zeros((1, 1))

        test.assert_allclose(x_hat, x_start)
        test.assert_allclose(feedback, exp_feedback)
        test.assert_allclose(feedforward, exp_feedfor)

    def test_quadratize_cost(self, Q, R):
        # inputs
        baseELQR = base.BaseELQR(Q=Q, R=R)
        x_start = np.array([[1], [2]])
        x_hat = np.arange(0, 2).reshape((2, 1))
        u_hat = np.array([2]).reshape((1, 1))
        u_nom = np.array([[1]])
        t = 0

        # expected outputs
        exp_P = np.zeros((1, 2))
        exp_q = np.array([[-0.00100000000000000],
                          [-0.00200000000000000]])
        exp_r = np.array([-0.100000000000000]).reshape((1, 1))

        # test function
        P, Q, R, q, r = baseELQR.quadratize_cost(x_hat, u_hat, timestep=t,
                                                 x_start=x_start,
                                                 u_nom=u_nom)

        # check
        test.assert_allclose(P, exp_P)
        test.assert_allclose(q, exp_q)
        test.assert_allclose(r, exp_r)

    def test_cost_to_go(self, Q, R):
        # inputs
        baseELQR = base.BaseELQR(Q=Q, R=R)
        cost_mat = np.eye(2)
        cost_vec = np.ones(2).reshape((2, 1))
        P = np.zeros((1, 2))
        q = np.zeros((2, 1))
        r = np.zeros((1, 1))
        A = np.eye(2)
        B = np.arange(0, 2).reshape((2, 1))
        c = np.zeros((2, 1))

        # expected outputs
        exp_feedback = np.array([[0, -0.909090909090909]])
        exp_feedfor = np.array([[-0.909090909090909]])
        exp_cost_mat = np.array([[1.00100000000000, 0],
                                 [0, 0.0919090909090908]])
        exp_cost_vec = np.array([[1],
                                 [0.0909090909090909]])

        # test function
        (cost_go_mat_out, cost_go_vec_out, feedback,
         feedforward) = baseELQR.cost_to_go(cost_mat, cost_vec, P, Q, R, q, r,
                                            A, B, c)

        # checking
        test.assert_allclose(cost_go_mat_out, exp_cost_mat)
        test.assert_allclose(cost_go_vec_out, exp_cost_vec)
        test.assert_allclose(feedback, exp_feedback)
        test.assert_allclose(feedforward, exp_feedfor)

    def test_cost_to_come(self, Q, R):
        # inputs
        baseELQR = base.BaseELQR(Q=Q, R=R)
        cost_mat = np.eye(2)
        cost_vec = np.ones(2).reshape((2, 1))
        P = np.zeros((1, 2))
        q = np.zeros((2, 1))
        r = np.zeros((1, 1))
        A = np.eye(2)
        B = np.arange(0, 2).reshape((2, 1))
        c = np.zeros((2, 1))

        # expected outputs
        exp_feedback = np.array([[0, -0.909173478655767]])
        exp_feedfor = np.array([[-0.908265213442325]])
        exp_cost_mat = np.array([[1.00100000000000, 0],
                                 [0, 0.0909173478655768]])
        exp_cost_vec = np.array([[1],
                                 [0.0908265213442326]])

        # test function
        (cost_come_mat_out, cost_come_vec_out, feedback,
         feedforward) = baseELQR.cost_to_come(cost_mat, cost_vec, P, Q, R, q,
                                              r, A, B, c)

        # checking
        test.assert_allclose(cost_come_mat_out, exp_cost_mat)
        test.assert_allclose(cost_come_vec_out, exp_cost_vec)
        test.assert_allclose(feedback, exp_feedback)
        test.assert_allclose(feedforward, exp_feedfor)

    def test_forward_pass(self, Q, R, func_list, inv_func_list):
        # inputs
        baseELQR = base.BaseELQR(Q=Q, R=R)
        x_hat = np.arange(0, 2).reshape((2, 1))
        x_start = x_hat.copy()
        u_nom = np.array([[1]])
        feedback = [np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2))]
        feedforward = [np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))]
        cost_come_mat = [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))]
        cost_come_vec = [np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1))]
        cost_go_mat = [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))]
        cost_go_vec = [np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1))]

        # expected outputs
        exp_x_hat = np.array([[-4.01398428760128],
                              [3.00282660925714]])
        exp_u_hat = np.array([[-0.000568845876679269]])

        fb1 = np.array([[-0.016913876239709, 0.253970455953737]])
        fb2 = np.array([[-0.542269168713002, -0.045531860887841]])
        fb3 = np.array([[1, 1]])
        exp_feedback = [fb1, fb2, fb3]

        ff1 = np.array([[2.896501661712843]])
        ff2 = np.array([[-1.196800257434532]])
        ff3 = np.array([[1]])
        exp_feedfor = [ff1, ff2, ff3]

        ccm1 = np.array([[1, 1],
                         [1, 1]])
        ccm2 = np.array([[0.000086061554319, -0.000444362293783],
                         [-0.000444362293783, 0.006460351960813]])
        ccm3 = np.array([[0.038599896953807, 0.003204216674860],
                         [0.003204216674858, 0.045756839785399]])
        exp_cost_come_mat = [ccm1, ccm2, ccm3]

        ccv1 = np.array([[1],
                         [1]])
        ccv2 = np.array([[-0.003036065006031],
                         [0.048171557708770]])
        ccv3 = np.array([[0.156475351126654],
                         [-0.113380502332024]])
        exp_cost_come_vec = [ccv1, ccv2, ccv3]

        # test function
        (x_hat, u_hat, feedback, feedforward, cost_come_mat,
         cost_come_vec) = baseELQR.forard_pass(x_hat, feedback,
                                               feedforward, cost_come_mat,
                                               cost_come_vec, cost_go_mat,
                                               cost_go_vec,
                                               dynamics=func_list,
                                               inverse_dynamics=inv_func_list,
                                               x_start=x_start, u_nom=u_nom)

        # checking
        test.assert_allclose(x_hat, exp_x_hat)
        test.assert_allclose(u_hat, exp_u_hat)
        for ii in range(0, len(exp_feedback)):
            test.assert_allclose(feedback[ii], exp_feedback[ii])
        for ii in range(0, len(exp_feedfor)):
            test.assert_allclose(feedforward[ii], exp_feedfor[ii])
        for ii in range(0, len(exp_cost_come_mat)):
            test.assert_allclose(cost_come_mat[ii], exp_cost_come_mat[ii])
        for ii in range(0, len(exp_cost_come_vec)):
            test.assert_allclose(cost_come_vec[ii], exp_cost_come_vec[ii])

    def test_backward_pass(self, Q, R, func_list, inv_func_list):
        # Inputs
        baseELQR = base.BaseELQR(Q=Q, R=R)
        x_hat = np.arange(0, 2).reshape((2, 1))
        x_start = x_hat.copy()
        u_nom = np.array([[1]])
        feedback = [np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2))]
        feedforward = [np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))]
        cost_come_mat = [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))]
        cost_come_vec = [np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1))]
        cost_go_mat = [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))]
        cost_go_vec = [np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1))]

        # Expected Values
        exp_x_hat = np.array([[1.14834458419351],
                              [-2.14621103302943]])
        exp_u_hat = np.array([[-0.00172699202296389]])

        fb1 = np.array([[-0.847207594209788, -0.014217870424957]])
        fb2 = np.array([[0.293040293386971, -0.131868131182320]])
        fb3 = np.array([[1, 1]])
        exp_feedback = [fb1, fb2, fb3]

        ff1 = np.array([[1.945874951749496]])
        ff2 = np.array([[3.849816845062269]])
        ff3 = np.array([[1]])
        exp_feedfor = [ff1, ff2, ff3]

        cgm1 = np.array([[0.816130428667278, 0.013695256827117],
                         [0.013695256827117, 0.001419152006663]])
        cgm2 = np.array([[0.009682675370057, -0.003907203891583],
                         [-0.003907203891583, 0.002758241739988]])
        cgm3 = np.array([[1, 1],
                         [1, 1]])
        exp_cost_go_mat = [cgm1, cgm2, cgm3]

        cgv1 = np.array([[-0.909939597617146],
                         [-0.014814625476393]])
        cgv2 = np.array([[0.084439017733033],
                         [-0.037997557737296]])
        cgv3 = np.array([[1],
                         [1]])
        exp_cost_go_vec = [cgv1, cgv2, cgv3]

        # Test Function
        (x_hat, u_hat, feedback, feedforward, cost_come_mat,
         cost_come_vec) = baseELQR.backward_pass(x_hat, feedback,
                                                 feedforward, cost_come_mat,
                                                 cost_come_vec, cost_go_mat,
                                                 cost_go_vec,
                                                 dynamics=func_list,
                                                 inverse_dynamics=inv_func_list,
                                                 x_start=x_start, u_nom=u_nom)

        # checking
        test.assert_allclose(x_hat, exp_x_hat)
        test.assert_allclose(u_hat, exp_u_hat)
        for ii in range(0, len(exp_feedback)):
            test.assert_allclose(feedback[ii], exp_feedback[ii])
        for ii in range(0, len(exp_feedfor)):
            test.assert_allclose(feedforward[ii], exp_feedfor[ii])
        for ii in range(0, len(exp_cost_go_mat)):
            test.assert_allclose(cost_go_mat[ii], exp_cost_go_mat[ii])
        for ii in range(0, len(exp_cost_go_vec)):
            test.assert_allclose(cost_go_vec[ii], exp_cost_go_vec[ii])


class TestDensityBased:
    def test_constructor(self):
        densityBased = base.DensityBased()
        assert len(densityBased.targets.means) == 0

        exp_safety_factor = 2
        exp_y_ref = 0.8
        densityBased = base.DensityBased(safety_factor=exp_safety_factor,
                                         y_ref=exp_y_ref)
        test.assert_allclose(densityBased.safety_factor, exp_safety_factor)
        test.assert_allclose(densityBased.y_ref, exp_y_ref)

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

        densityBased.convert_waypoints(waypoints)

        for ii, cov in enumerate(densityBased.targets.covariances):
            test.assert_allclose(cov, exp_covs[ii],
                                 err_msg='iteration {}'.format(ii))

    def test_update_targets(self):
        assert 0, 'implement'
