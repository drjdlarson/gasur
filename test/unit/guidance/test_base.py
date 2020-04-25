# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:24:30 2020

@author: ryan4
"""
import pytest
import numpy as np
import numpy.testing as test

from gasur.guidance.base import BaseLQR, BaseELQR


@pytest.mark.incremental
class TestBaseLQR:
    def test_constructor(self, Q, R):
        baseLQR = BaseLQR(Q=Q, R=R)
        test.assert_allclose(baseLQR.state_penalty, Q)
        test.assert_allclose(baseLQR.ctrl_penalty, R)

    def test_iterate(self, Q, R):
        baseLQR = BaseLQR(Q=Q, R=R)
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
        baseELQR = BaseELQR(Q=Q, R=R)
        test.assert_allclose(baseELQR.state_penalty, Q)
        test.assert_allclose(baseELQR.ctrl_penalty, R)

        mi = 20
        baseELQR = BaseELQR(Q=Q, R=R, max_iters=mi)
        test.assert_allclose(baseELQR.state_penalty, Q)
        test.assert_allclose(baseELQR.ctrl_penalty, R)

        assert baseELQR.max_iters == mi

    def test_initialize(self, Q, R):
        baseELQR = BaseELQR(Q=Q, R=R)
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
        baseELQR = BaseELQR(Q=Q, R=R)
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
        baseELQR = BaseELQR(Q=Q, R=R)
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
        baseELQR = BaseELQR(Q=Q, R=R)
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

    def test_forward_pass(self):
        assert 0, 'implement'

    def test_backward_pass(self):
        assert 0, 'implement'
