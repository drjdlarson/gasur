# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:18:28 2020

@author: ryan4
"""
import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvnpdf

from ..estimator import GaussianMixture
from gasur.utilities.math import get_state_jacobian, get_input_jacobian


class BaseLQR:
    def __init__(self, **kwargs):
        self.state_penalty = kwargs['Q']
        self.ctrl_penalty = kwargs['R']
        def_rows = self.ctrl_penalty.shape[1]
        def_cols = self.state_penalty.shape[0]
        self.cross_penalty = kwargs.get('corss_penalty',
                                        np.zeros((def_rows, def_cols)))

    def iterate(self, **kwargs):
        # process input arguments
        F = kwargs['F']
        del kwargs['F']
        G = kwargs['G']
        del kwargs['G']
        total_time_steps = kwargs.get('total_time_steps', None)
        inf_horizon = total_time_steps is None

        if inf_horizon:
            P = la.solve_discrete_are(F, G, self.state_penalty,
                                      self.ctrl_penalty)
            feedback_gain = la.inv(G.T @ P @ G + self.ctrl_penalty) \
                @ (G.T @ P @ F + self.cross_penalty)
            return feedback_gain
        else:
            # ##TODO: implement
            c = self.__class__.__name__
            name = self.iterate.__name__
            msg = '{}.{} not implemented'.format(c, name)
            raise RuntimeError(msg)


class BaseELQR(BaseLQR):
    def __init__(self, **kwargs):
        self.max_iters = kwargs.get('max_iters', 50)
        super().__init__(**kwargs)

    def initialize(self, x_start, n_inputs, **kwargs):
        n_states = x_start.size
        x_hat = x_start
        feedback = kwargs.get('feedback', np.zeros((n_inputs, n_states)))
        feedforward = kwargs.get('feedforward', np.zeros((n_inputs, 1)))
        cost_go_mat = kwargs.get('cost_go_mat', np.zeros((n_states, n_states)))
        cost_go_vec = kwargs.get('cost_go_vec', np.zeros((n_states, 1)))
        cost_come_mat = kwargs.get('cost_come_mat',
                                   np.zeros((n_states, n_states)))
        cost_come_vec = kwargs.get('cost_come_vec', np.zeros((n_states, 1)))
        return (feedback, feedforward, cost_go_mat, cost_go_vec,
                cost_come_mat, cost_come_vec, x_hat)

    def quadratize_cost(self, x_hat, u_hat, **kwargs):
        '''
        This assumes the true cost function is given by:
            c_0 = 1/2(x - x_0)^T Q (x - x_0) + 1/2(u - u_0)^T R (u - u_0)
            c_t = 1/2(u - u_nom)^T R (u - u_nom) + non quadratic state term(s)
            c_end = 1/2(x - x_end)^T Q (x - x_end)
        '''
        timestep = kwargs['timestep']
        x_start = kwargs['x_start']
        u_nom = kwargs['u_nom']
        if timestep == 0:
            Q = self.state_penalty
            q = -Q @ x_start
        else:
            Q, q = self.quadratize_non_quad_state(x_hat, u_hat, **kwargs)

        R = self.ctrl_penalty
        r = -R @ u_nom
        P = np.zeros((u_nom.size, x_start.size))

        return P, Q, R, q, r

    def quadratize_non_quad_state(self, x_hat, u_hat, **kwargs):
        Q = self.state_penalty
        q = np.zeros((x_hat.size, 1))
        return Q, q

    def quadratize_final_cost(self, x_hat, u_hat, **kwargs):
        '''
        This assumes the true cost function is given by:
            c_0 = 1/2(x - x_0)^T Q (x - x_0) + 1/2(u - u_0)^T R (u - u_0)
            c_t = 1/2(u - u_nom)^T R (u - u_nom) + non quadratic state term(s)
            c_end = 1/2(x - x_end)^T Q (x - x_end)
        '''
        x_end = kwargs['x_end']
        Q = self.state_penalty
        q = -Q @ x_end

        return Q, q

    def iterate(self, x_start, x_end, u_nom, **kwargs):
        cost_function = kwargs['cost_function']
        final_cost_function = kwargs['final_cost_function']
        dynamics_fncs = kwargs['dynamics_fncs']

        feedback, feedforward, cost_go_mat, cost_go_vec, cost_come_mat, \
            cost_come_vec, x_hat = self.initialize(x_start, u_nom.size,
                                                   **kwargs)

        converged = False
        old_cost = 0
        for iteration in range(0, self.max_iters):
            # forward pass
            x_hat, u_hat, feedback, feedforward, cost_come_mat, \
                cost_come_vec = self.forard_pass(x_hat, feedback, feedforward,
                                                 cost_come_mat,
                                                 cost_come_vec, cost_go_mat,
                                                 cost_go_vec, x_start=x_start,
                                                 u_nom=u_nom, **kwargs)

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

    def cost_to_go(self, cost_go_mat, cost_go_vec, P, Q, R, q, r,
                   state_mat, input_mat, c_vec, **kwargs):
        c_mat = input_mat.T @ cost_go_mat @ state_mat + P
        d_mat = state_mat.T @ cost_go_mat @ state_mat + Q
        e_mat = input_mat.T @ cost_go_mat @ input_mat + R
        tmp = (cost_go_vec + cost_go_mat @ c_vec)
        d_vec = state_mat.T @ tmp + q
        e_vec = input_mat.T @ tmp + r

        e_inv = la.inv(e_mat)
        feedback = -e_inv @ c_mat
        feedforward = -e_inv @ e_vec

        cost_go_mat_out = d_mat + c_mat.T @ feedback
        cost_go_vec_out = d_vec + c_mat.T @ feedforward

        return cost_go_mat_out, cost_go_vec_out, feedback, feedforward

    def cost_to_come(self, cost_come_mat, cost_come_vec, P, Q, R, q, r,
                     state_mat_bar, input_mat_bar, c_bar_vec, **kwargs):
        S_bar_Q = cost_come_mat + Q
        s_bar_q_sqr_c_bar = cost_come_vec + q + S_bar_Q @ c_bar_vec

        c_bar_mat = input_mat_bar.T @ S_bar_Q @ state_mat_bar \
            + P @ state_mat_bar
        d_bar_mat = state_mat_bar.T @ S_bar_Q @ state_mat_bar
        e_bar_mat = input_mat_bar.T @ S_bar_Q @ input_mat_bar \
            + R + P @ input_mat_bar + input_mat_bar.T @ P.T
        d_bar_vec = state_mat_bar.T @ s_bar_q_sqr_c_bar
        e_bar_vec = input_mat_bar.T @ s_bar_q_sqr_c_bar + r + P @ c_bar_vec

        # find controller gains
        e_inv = la.inv(e_bar_mat)
        feedback = -e_inv @ c_bar_mat
        feedforward = -e_inv @ e_bar_vec

        # update cost-to-come
        cost_come_mat_out = d_bar_mat + c_bar_mat.T @ feedback
        cost_come_vec_out = d_bar_vec + c_bar_mat.T @ feedforward

        return cost_come_mat_out, cost_come_vec_out, feedback, feedforward

    def backward_pass(self, x_hat, feedback, feedforward,
                      cost_come_mat, cost_come_vec, cost_go_mat, cost_go_vec,
                      **kwargs):
        dynamics_fncs = kwargs['dynamics']
        inv_dynamics_fncs = kwargs['inverse_dynamics']

        max_time_steps = len(cost_come_mat)
        for kk in range(max_time_steps - 2, -1, -1):
            u_hat = feedback[kk] @ x_hat + feedforward[kk]
            x_hat_prime = np.zeros(x_hat.shape)
            for ii, gg in enumerate(inv_dynamics_fncs):
                x_hat_prime[ii] = gg(x_hat, u_hat, **kwargs)

            state_mat = get_state_jacobian(x_hat_prime, u_hat,
                                           dynamics_fncs, **kwargs)
            input_mat = get_input_jacobian(x_hat_prime, u_hat,
                                           dynamics_fncs, **kwargs)
            c_vec = x_hat - state_mat @ x_hat_prime - input_mat @ u_hat

            P, Q, R, q, r = self.quadratize_cost(x_hat_prime, u_hat,
                                                 timestep=kk, **kwargs)

            (cost_go_mat[kk], cost_go_vec[kk], feedback[kk],
             feedforward[kk]) = self.cost_to_go(cost_go_mat[kk+1],
                                                cost_go_vec[kk+1],
                                                P, Q, R, q, r, state_mat,
                                                input_mat, c_vec)

            # update state estimate
            x_hat = -la.inv(cost_go_mat[kk] + cost_come_mat[kk]) \
                @ (cost_go_vec[kk] + cost_come_vec[kk])

        return (x_hat, u_hat, feedback, feedforward, cost_come_mat,
                cost_come_vec)

    def forard_pass(self, x_hat, feedback, feedforward, cost_come_mat,
                    cost_come_vec, cost_go_mat, cost_go_vec, **kwargs):
        dynamics_fncs = kwargs['dynamics']
        inv_dynamics_fncs = kwargs['inverse_dynamics']

        max_time_steps = len(cost_come_mat)
        for kk in range(0, max_time_steps - 1):
            u_hat = feedback[kk] @ x_hat + feedforward[kk]
            x_hat_prime = np.zeros(x_hat.shape)
            for ii, ff in enumerate(dynamics_fncs):
                x_hat_prime[ii] = ff(x_hat, u_hat, **kwargs)

            state_mat_bar = get_state_jacobian(x_hat_prime, u_hat,
                                               inv_dynamics_fncs, **kwargs)
            input_mat_bar = get_input_jacobian(x_hat_prime, u_hat,
                                               inv_dynamics_fncs, **kwargs)
            c_bar_vec = x_hat - state_mat_bar @ x_hat_prime \
                - input_mat_bar @ u_hat

            P, Q, R, q, r = self.quadratize_cost(x_hat, u_hat, timestep=kk,
                                                 **kwargs)

            (cost_come_mat[kk+1], cost_come_vec[kk+1], feedback[kk],
             feedforward[kk]) = self.cost_to_come(cost_come_mat[kk],
                                                  cost_come_vec[kk],
                                                  P, Q, R, q, r, state_mat_bar,
                                                  input_mat_bar, c_bar_vec)

            # update state estimate
            x_hat = -la.inv(cost_go_mat[kk+1] + cost_come_mat[kk+1]) \
                @ (cost_go_vec[kk+1] + cost_come_vec[kk+1])

        return (x_hat, u_hat, feedback, feedforward, cost_come_mat,
                cost_come_vec)


class DensityBased:
    def __init__(self, wayareas=[], saftey_factor=1, y_ref=0.9):
        self.targets = wayareas
        self.saftey_factor = saftey_factor
        self.y_ref = y_ref

    def density_based_cost(self, **kwargs):
        key = 'obj_weights'
        obj_weights = kwargs[key]
        del kwargs[key]
        key = 'obj_states'
        obj_states = kwargs[key]
        del kwargs[key]
        key = 'obj_covariances'
        obj_covariances = kwargs[key]
        del kwargs[key]

        target_center = self.target_center()
        num_targets = len(self.targets.means)
        num_objects = obj_states.shape[1]

        # find radius of influence and shift
        max_dist = 0
        for tar_mean in self.targets.means:
            diff = tar_mean - target_center
            dist = np.sqrt(diff.transpose() @ diff)
            if dist > max_dist:
                max_dist = dist
        radius_of_influence = self.saftey_factor * max_dist
        shift = -1 * np.log(1/self.y_ref - radius_of_influence)

        # get actiavation term
        max_dist = 0
        for ii in range(0, num_objects):
            diff = target_center - obj_states[:, [ii]]
            dist = np.sqrt(diff.transpose @ diff)
            if dist > max_dist:
                max_dist = dist
        activator = 1 / (1 + np.exp(-max_dist - shift))

        # get maximum stand
        max_var_obj = np.max(np.diag(obj_covariances))
        max_var_target = 0
        for cov in self.targets.covariances:
            var = np.max(np.diag(cov))
            if var > max_var_target:
                max_var_target = var

        # Loop for all double summation terms
        sum_obj_obj = 0
        sum_obj_target = 0
        sum_target_target = 0
        quad = 0
        for outer_obj in range(0, num_objects):
            # object to object
            for inner_obj in range(0, num_objects):
                comb_cov = obj_covariances[outer_obj] \
                            + obj_covariances[inner_obj]
                sum_obj_obj += obj_weights[outer_obj] \
                    * obj_weights[inner_obj] \
                    * mvnpdf.pdf(obj_states[outer_obj],
                                 mean=obj_states[inner_obj],
                                 covariance=comb_cov)

            # object to target and quadratic
            for ii in range(0, num_targets):
                # object to target
                comb_cov = obj_covariances[outer_obj] \
                    + self.targets.covariances[ii]
                sum_obj_target += obj_weights[outer_obj] \
                    * self.targets.weights[ii] \
                    * mvnpdf.pdf(obj_states[outer_obj],
                                 mean=self.targets.means[ii],
                                 covariance=comb_cov)

                # quadratic
                diff = obj_states[outer_obj] - self.targets.means[ii]
                log_term = np.log((2*np.pi)**(-0.5*num_targets) *
                                  la.det(comb_cov)**-0.5) \
                    - 0.5 * diff.transpose() @ la.inv(comb_cov) @ diff
                quad += obj_weights[outer_obj] * self.targets.weights[ii] \
                    * log_term

        # target to target
        for outer in range(0, num_targets):
            for inner in range(0, num_targets):
                comb_cov = self.targets.covariances[outer] \
                    + self.targets.covariances[inner]
                sum_target_target += self.targets.weights[outer] \
                    * self.targets.weights[inner] \
                    * mvnpdf.pdf(self.targets.means[outer],
                                 mean=self.targets.means[inner],
                                 covariance=comb_cov)

        return 10 * num_objects * max_var_obj * (sum_obj_obj - 20
                                                 * max_var_target * num_targets
                                                 * sum_obj_target) \
            + sum_target_target + activator * quad

    def convert_waypoints(self, **kwargs):
        waypoints = kwargs['waypoints']
        del kwargs['waypoints']
        center_len = waypoints[0].size
        num_waypoints = len(waypoints)
        combined_centers = np.zeros((center_len, num_waypoints+1))

        # get overall center, build collection of centers
        for ii in range(0, num_waypoints):
            combined_centers[:, [ii]] = waypoints[ii].reshape((center_len, 1))
            combined_centers[:, [-1]] += combined_centers[:, [ii]]
        combined_centers[:, [-1]] = combined_centers[:, [-1]] / num_waypoints

        # find directions to each point
        directions = np.zeros((center_len, num_waypoints + 1, num_waypoints
                               + 1))
        for start_point in range(0, num_waypoints + 1):
            for end_point in range(0, num_waypoints + 1):
                directions[:, start_point, end_point] = \
                    (combined_centers[:, end_point]
                     - combined_centers[:, start_point]).reshape(center_len, 1,
                                                                 1)

        def find_principal_components(data):
            num_samps = data.shape[0]
            num_feats = data.shape[1]

            mean = np.sum(data, 1) / num_samps
            covars = np.zeros((num_feats, num_feats))
            for ii in range(0, num_feats):
                for jj in range(0, num_feats):
                    acc = 0
                    for samp in range(0, num_samps):
                        acc += (data[samp, ii] - mean[ii]) \
                                * (data[samp, jj] - mean[jj])
                    covars[ii, jj] = acc / num_samps
            (w, comps) = la.eig(covars)
            return comps.T()

        def find_largest_proj_dist(new_dirs, old_dirs):
            vals = np.zeros(new_dirs.shape[0])
            for ii in range(0, new_dirs.shape[1]):
                for jj in range(0, old_dirs.shape[1]):
                    proj = np.abs(new_dirs[:, [ii]].T @ old_dirs[:, [jj]])
                    if proj > vals[ii]:
                        vals[ii] = proj
            return np.diag(vals)

        weight = 1 / num_waypoints
        wayareas = GaussianMixture()
        for wp_ind in range(0, num_waypoints):
            center = waypoints[wp_ind].reshape((center_len, 1))

            sample_data = combined_centers
            sample_data = np.delete(sample_data, wp_ind, 1)
            sample_dirs = directions[:, wp_ind, :].squeeze()
            sample_dirs = np.delete(sample_dirs, wp_ind, 1)
            comps = find_principal_components(sample_data.T)
            vals = find_largest_proj_dist(comps, sample_dirs)

            covariance = comps @ vals @ la.inv(comps)

            wayareas.means.append(center)
            wayareas.covariances.append(covariance)
            wayareas.weight.append(weight)
        return wayareas

    def update_targets(self, **kwargs):
        key = 'new_waypoints'
        new_waypoints = kwargs[key]
        del kwargs[key]
        reset = kwargs.get('reset', False)
        if not reset:
            for m in self.wayareas.means:
                new_waypoints.append(m)

        # clear way areas and add new ones
        self.wayareas = GaussianMixture()
        self.wayareas = self.convert_waypoints(new_waypoints)

    def target_center(self):
        summed = np.zeros(self.targets.means[0].shape)
        num_tars = len(self.targets.means)
        for ii in range(0, num_tars):
            summed += self.targets.means[ii]
        return summed / num_tars


class GaussianObject:
    def __init__(self, **kwargs):
        self.dyn_functions = kwargs.get('dyn_functions', [])
        self.inv_dyn_functions = kwargs.get('inv_dyn_functions', [])

        # each timestep is a row
        self.means = kwargs.get('means', np.array([[]]))
        self.ctrl_input = kwargs('control_input', np.array([[]]))

        # lists of arrays
        self.feedforward_lst = kwargs.get('feedforward', [])
        self.feedback_lst = kwargs.get('feedback', [])
        self.cost_to_come_mat = kwargs.get('cost_to_come_mat', [])
        self.cost_to_come_vec = kwargs.get('cost_to_come_vec', [])
        self.cost_to_go_mat = kwargs.get('cost_to_go_mat', [])
        self.cost-to_go_vec = kwargs.get('cost_go_vec', [])

        # only 1 covariance for entire trajectory
        self.covariance = kwargs.get('covariance', np.array([[]]))
