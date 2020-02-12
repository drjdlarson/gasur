# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:20:30 2020

@author: ryan4
"""
import numpy as np
from scipy import linalg

from .base import DensityBasedGuidance
from ..utilities.math import get_state_space_derivatives
from ..enumerations import GuidanceType


class DensityBasedExtendedLQR(DensityBasedGuidance):
    def __init__(self):
        super().__init__(guidancetype=GuidanceType.ELQR)
        self.q_matrix = 0
        self.r_matrix = 0
        self.nominal_inputs = 0
        self.convergence_threshold = 0.1
        self.max_iterations = 10

    def reinitialize_trajectories(self):
        pass

    def update(self, interpreter, dt):
        states, inputs, dyn_fnc_lst, inv_dyn_fnc_lst \
            = self.reinitialize_trajectories(interpreter)
        state_0 = states[:, :, 1].squeeze()
        state_dims = states.shape
        input_dims = inputs.shape

        input_len = input_dims[0]
        state_len = state_dims[0]
        number_obj = state_dims[1]

        iteration = 0
        converged = False
        while iteration < self.max_iterations and (not converged):
            iteration += 1

            # Forward LQR pass
            guidance_inputs = np.zeros((input_len, number_obj))
            for t in range(0, self.time_horizon_steps-1):

                # propagate guidance input for each object
                for obj in range(0, number_obj):
                    guidance_inputs[:, obj] = np.squeeze(
                                   feedback_gain[:, :, obj]) \
                            @ np.squeeze(states[:, obj, t]) \
                            + feedforward_gain[:, obj]

                # perform LQR for each object at current time
                for obj in range(0, number_obj):
                    # propogate dynamics of current object
                    next_state = dyn_fnc_lst[obj](states[:, obj, t].squeeze(),
                                                  guidance_inputs[:, obj],
                                                  dt)

                    # linearize inverse dynamics about propagated state
                    A_bar, B_bar, c_vec_bar = get_state_space_derivatives(
                            inv_dyn_fnc_lst[obj], next_state,
                            guidance_inputs[:, obj], )

                    # quadratize cost function about current object and frozen
                    # state
                    Q, R, P, q_vec, r_vec \
                        = self.quadratize_cost(states[:, t, :].squeeze(),
                                               guidance_inputs, t, state_0,
                                               interpreter.gaussian_weights,
                                               interpreter.gaussian_covariances,
                                               obj)

                    # calculate guidance update for current object
                    s_q_sum = (S_bar[:, :, obj, t].squeeze() + Q)
                    C_bar = (B_bar.T @ s_q_sum + P) @ A_bar
                    D_bar = A_bar.T @ s_q_sum @ A_bar
                    E_bar = B_bar.T @ s_q_sum @ B_bar + R + P @ B_bar \
                        + B_bar.T @ P.T
                    s_q_vec_sum = s_vec_bar[:, obj, t] + q_vec + s_q_sum \
                        @ c_vec_bar
                    d_vec_bar = A_bar @ s_q_vec_sum
                    e_vec_bar = r_vec + P @ c_vec_bar + B_bar.T @ s_q_vec_sum

                    E_inv = linalg.inv(E_bar)
                    feedback_gain[:, :, obj] = E_inv @ C_bar
                    feedforward_gain[:, obj] = E_inv @ e_vec_bar

                    # update cost-to-come functions
                    
                    # update next state for current object
                    

            # Update final cost for each object
            for obj in range(0, number_obj):
                # find goal state based on closest target to current object
                
                # update cost-to-go functions
                
                # update final state
                

            # Backward LQR pass
            for t in range(self.time_horizon_steps, 0, -1):
                # calculate control and backpropagate states
                for obj in range(0, number_obj):
                    
                    
                # perform backward LQR for each object
                for obj in range(0, number_obj):
                    # linearize dynamics about propagated state
                    
                    # quadratize cost function about current object and 
                    # propagated states
                    
                    # calculate guidance update for current object
                    
                    # update cost-to-go functions
                    
                    # update state for current object
                    
                    
            # Find cost of swarm trajectory
            cost = 0
            for t in range(0, self.time_horizon_steps):
                # update control inputs
                
                # update cost
                cost += 0

                # propagate states
                for obj in range(0, number_obj):
                    
                    
            converged = abs(cost - last_cost) < self.convergence_threshold
            last_cost = cost
            
            # save trajectories, guidance gains, cost-to-go, and cost-to come
            # for reinitialization next guidance update
            

    def quadratize_cost(self, states, guidance_inputs, time_step, state_0,
                        gaussian_weights, gaussian_covariances, guassian_ind):
        # TODO: define this function
        cost_jacobian, cost_hessian = get_cost_jacobian_hessian()

        # get sizes
        state_len = states.shape[0]
        guide_len = guidance_inputs.shape[0]

        # define Q matrix based on time step
        if time_step == 0:
            Q = self.q_matrix
        else:
            Q = cost_hessian

        # define P and R matrices
        P = cost_hessian
        R = self.r_matrix

        # define q_vec
        if time_step == 0:
            q_vec = -self.q_matrix @ state_0[:, gaussian_ind]
        else:
            # build jacobian for single object
            single_jacobian = np.zeros((state_len + guide_len, 1))
            single_jacobian[0:state_len] = cost_jacobian
            single_jacobian[state_len:] = cost_jacobian

            combined_mat = np.vstack((np.hstack((Q, P.T)), np.hstack((P, R))))
            combined_vec = single_jacobian - combined_mat \
                @ np.vstack((states[:, gaussian_ind],
                             guidance_inputs[:, gaussian_ind]))
            q_vec = -cost_hessian @ states[:, gaussian_ind] \
                + combined_vec[1:state_len]

        r_vec = self.r_matrix @ self.nominal_inputs[:, gaussian_ind0]

        return (Q, R, P, q_vec, r_vec)
