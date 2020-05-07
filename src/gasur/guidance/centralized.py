# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:22:52 2020

@author: ryan4
"""
import numpy as np
import scipy.linalg as la

from gasur.guidance.base import BaseELQR, DensityBased
from gasur.utilities.math import get_state_jacobian, get_input_jacobian, \
    get_hessian, get_jacobian


class ELQRGaussian(BaseELQR, DensityBased):
    def __init__(self, cur_gaussians=None, **kwargs):
        if cur_gaussians is None:
            cur_gaussians = []
        self.gaussians = cur_gaussians  # list of GaussianObjects
        super().__init__(**kwargs)

    def fowrward_pass(self, **kwargs):
        max_time_steps = self.gaussians.means.shape[0]
        num_gaussians = len(self.gaussians)
        for kk in range(0, max_time_steps - 1):
            cur_states = np.zeros((num_gaussians,
                                   self.gaussians.means.shape[1]))
            for ii, gg in enumerate(self.gaussians):
                cur_states[ii, :] = gg.means[kk, :]

            # update control for each gaussian
            for gg in self.gaussians:
                gg.ctrl_input[kk, :] = (gg.feedback[kk].dot(gg.means[kk, :])
                                        + gg.feedforward[kk])

            # update values
            for gg in self.gaussians:
                next_state = np.zeros((self.gaussians.means.shape[1], 1))
                for ii, ff in enumerate(gg.dyn_functions):
                    next_state[ii] = ff(gg.means[kk, :], gg.ctrl_input[kk, :],
                                        **kwargs)

                state_mat_bar = get_state_jacobian(next_state,
                                                   gg.ctrl_input[kk, :],
                                                   gg.inv_dyn_functions,
                                                   **kwargs)
                input_mat_bar = get_input_jacobian(next_state,
                                                   gg.ctrl_input[kk, :],
                                                   gg.inv_dyn_functions,
                                                   **kwargs)
                c_bar_vec = gg.means[[kk], :].T - state_mat_bar @ next_state \
                    - input_mat_bar.dot(gg.ctrl_input[kk, :])

                # ##TODO: fix inputs
                P, Q, R, q, r = self.quadratize_cost(x_hat, u_hat, timestep=kk,
                                                 **kwargs)

                (gg.cost_come_mat[kk+1], gg.cost_come_vec[kk+1],
                 gg.feedback[kk],
                 gg.feedforward[kk]) = self.cost_to_come(gg.cost_come_mat[kk],
                                                         gg.cost_come_vec[kk],
                                                         P, Q, R, q, r,
                                                         state_mat_bar,
                                                         input_mat_bar,
                                                         c_bar_vec)

                # update state estimate
                gg.means[[kk+1], :] = (-la.inv(gg.cost_go_mat[kk+1]
                                               + gg.cost_come_mat[kk+1])
                                       @ (gg.cost_go_vec[kk+1]
                                          + gg.cost_come_vec[kk+1])).T

    def backward_pass(self, **kwargs):
        max_time_steps = self.gaussians.means.shape[0]
        num_gaussians = len(self.gaussians)
        for kk in range(max_time_steps - 2, -1, -1):
            # save values for each gaussian
            prev_states = np.zeros((num_gaussians,
                                   self.gaussians.means.shape[1]))
            for ii, gg in enumerate(self.gaussians):
                gg.ctrl_input[kk, :] = gg.feedback[kk].dot(gg.means[kk+1, :]) \
                    + gg.feedforward[kk]
                for jj, ff in enumerate(gg.inv_dyn_functions):
                    prev_states[ii, jj] = ff(gg.means[kk+1, :],
                                             gg.ctrl_input[kk, :], **kwargs)

            # update values
            for ii, gg in enumerate(self.gaussians):
                state_mat_bar = get_state_jacobian(prev_states[ii, :],
                                                   gg.ctrl_input[kk, :],
                                                   gg.dyn_functions,
                                                   **kwargs)
                input_mat_bar = get_input_jacobian(prev_states[ii, :],
                                                   gg.ctrl_input[kk, :],
                                                   gg.inv_dyn_functions,
                                                   **kwargs)
                c_bar_vec = gg.means[[kk], :].T - state_mat_bar \
                    @ prev_states[ii, :] \
                    - input_mat_bar.dot(gg.ctrl_input[kk, :])

                # ##TODO: fix inputs
                P, Q, R, q, r = self.quadratize_cost(x_hat, u_hat, timestep=kk,
                                                 **kwargs)

                (gg.cost_come_mat[kk], gg.cost_come_vec[kk],
                 gg.feedback[kk],
                 gg.feedforward[kk]) = self.cost_to_come(gg.cost_come_mat[kk+1],
                                                         gg.cost_come_vec[kk+1],
                                                         P, Q, R, q, r,
                                                         state_mat_bar,
                                                         input_mat_bar,
                                                         c_bar_vec)

                # update state estimate
                gg.means[[kk], :] = (-la.inv(gg.cost_go_mat[kk+1]
                                             + gg.cost_come_mat[kk+1])
                                     @ (gg.cost_go_vec[kk+1]
                                        + gg.cost_come_vec[kk+1])).T

    def quadratize_non_quad_state(self, all_states, obj_num, **kwargs):
        def helper(x, cur_states):
            loc_states = cur_states.copy()
            loc_states[:, [obj_num]] = x
            weight_lst = []
            cov_lst = []
            for ii in self.gaussians:
                weight_lst.append(ii.weight)
                cov_lst.append(ii.covariance)
            return self.density_based_cost(loc_states, weight_lst, cov_lst)

        Q = get_hessian(all_states[:, [obj_num]].copy(),
                        lambda x_: helper(x_, all_states), **kwargs)
        q = get_jacobian(all_states[:, [obj_num]].copy(),
                         lambda x_: helper(x_, all_states), **kwargs)

        # Regularize Matrix
        eig_vals, eig_vecs = la.eig(Q)
        for ii in range(0, eig_vals.size):
            if eig_vals[ii] < 0:
                eig_vals[ii] = 0
        Q = eig_vecs @ np.diag(eig_vals) @ la.inv(eig_vecs)

        return Q, q

    def quadratize_cost(self, all_states, obj_num, timestep, x_start, u_nom,
                        **kwargs):
        '''
        This assumes the true cost function is given by:
            c_0 = 1/2(x - x_0)^T Q (x - x_0) + 1/2(u - u_0)^T R (u - u_0)
            c_t = 1/2(u - u_nom)^T R (u - u_nom) + non quadratic state term(s)
            c_end = 1/2(x - x_end)^T Q (x - x_end)
        '''
        if timestep == 0:
            Q = self.state_penalty
            q = -Q @ x_start
        else:
            Q, q = self.quadratize_non_quad_state(all_states, obj_num,
                                                  **kwargs)

        R = self.ctrl_penalty
        r = -R @ u_nom
        P = np.zeros((u_nom.size, x_start.size))

        return P, Q, R, q, r

    def iterate(self, **kwargs):
        # ##TODO: implement
        msg = '{}.{} not implemented'.format(self.__class__.__name__,
                                             self.iterate.__name__)
        raise RuntimeError(msg)
        final_cost_function = kwargs['final_cost_function']
        dynamics_fncs = kwargs['dynamics_fncs']

        feedback, feedforward, cost_go_mat, cost_go_vec, cost_come_mat, \
            cost_come_vec, x_hat = self.initialize(x_start, u_nom.size,
                                                   **kwargs)

        converged = False
        old_cost = 0
        for iteration in range(0, self.max_iters):
            # forward pass
            self.forard_pass(x_start=x_start, u_nom=u_nom, **kwargs)

            # quadratize final cost
            cost_go_mat[-1], cost_go_vec[-1] = \
                self.quadratize_final_cost(x_hat, u_hat, x_end=x_end, **kwargs)
            x_hat = -la.inv(cost_go_mat[-1] + cost_come_mat[-1]) \
                @ (cost_go_vec[-1] + cost_come_vec[-1])

            # backward pass
            x_hat, u_hat, feedback, feedforward, cost_go_mat, \
                cost_go_vec = self.backward_pass(x_hat, feedback, feedforward,
                                                 cost_come_mat,
                                                 cost_come_vec, cost_go_mat,
                                                 cost_go_vec, x_start=x_start,
                                                 u_nom=u_nom, **kwargs)

            # find real cost of trajectory
            state = x_hat
            cur_cost = 0
            for kk in range(0, len(feedback)-1):
                ctrl_input = feedback[kk] @ state + feedforward[kk]
                cur_cost += cost_function(state, ctrl_input, **kwargs)
                state = dynamics_fncs(state, ctrl_input, **kwargs)
            cur_cost += final_cost_function(state, **kwargs)

            # check for convergence
            if iteration != 0:
                converged = np.abs(old_cost - cur_cost) < self.cost_tol

            old_cost = cur_cost
            if converged:
                break

        return feedback, feedforward
