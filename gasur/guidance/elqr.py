# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:20:30 2020

@author: ryan4
"""
import numpy as np

from .base import LQR
from ..utilities.math import get_state_space_derivatives
from ..enumerations import GuidanceType


class ExtendedLQR(LQR):
    def __init__(self):
        super().__init__(guidancetype=GuidanceType.ELQR)
        self.q_matrix = 0
        self.r_matrix = 0
        self.nominal_control = 0
        self.convergence_threshold = 0.1
        self.max_iterations = 10


    def reinitialize_trajectories(self):
        pass


    def update(self, interpreter, dt):
        states, inputs, dyn_fnc_lst, inv_dyn_fnc_lst = \
            self.reinitialize_trajectories(interpreter)
        state_dims = states.shape
        input_dims = interpreter.inputs.shape

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
                    next_state = dyn_fnc_lst[obj](state[:, [obj]],
                                                  guidance_inputs[:, [obj]],
                                                  dt)

                    # linearize inverse dynamics about propagated state
                    A_bar, B_bar = get_state_space_derivatives(
                            inv_dyn_fnc_lst[obj], next_state, 
                            guidance_inputs[:, [obj]], )

                    # quadratize cost function about current object and frozen
                    # state
                    
                    # calculate guidance update for current object
                    
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
            


    def quadratize_cost(self, states, guidance_input, time_step, state_0,
                        gaussian_weights, gaussian_covariances, guassian_ind):
        cost_jacobian, cost_hessian = get_cost_jacobian_hessian()

        if time_step == 0:
            Q = self.q_matrix
        else:
            Q = cost_hessian

        
        return
