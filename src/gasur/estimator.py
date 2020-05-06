# -*- coding: utf-8 -*-
"""
This file contains the relevant functions for the Estimator developed for the
Laboratori for Autonomous GNC and Estimation Research (LAGER) at the University
of Alabama.

Version 0.1
Author: Vaughn Weirens
"""

# %% Import relevant and necessary modules.
import numpy as np
import scipy

# %% Gaussian Mixture


class GaussianMixture:
    def __init__(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = weights

# %% General Estimator Class


class Estimator:
    def __init__(self, maximum_number_of_objects):
        self.maximum_number_of_objects = 100
        self.prune_threshold = 1e-5
        self.merge_threshold = 10
        self.gate_size_percent = 0.99
        self.gateFlag = 'off'
        self.weight_threshold = 0.5


# %% EKF - Still in progress
class ExtendedKalmanFilter(Estimator):
    def __init__(self):
        # ##TODO: implement
        msg = '{}.{} not implemented'.format(self.__class__.__name__,
                                             self.__init__.__name__)
        raise RuntimeError(msg)

    def manage(self):   
        # ##TODO: implement
        msg = '{}.{} not implemented'.format(self.__class__.__name__,
                                             self.manage.__name__)
        raise RuntimeError(msg)

    def gate(simulation_parameters, gnc_system,):
        valid_index_array = np.array([])
        if gnc_system.number_measurements == 0:
            gnc_system.measurements = np.array([])
            
        for ii in range(0, number_gaussians_prediction):
            continue
        # ##TODO: implement
        msg = '{}.{} not implemented'.format(self.__class__.__name__,
                                             self.gate.__name__)
        raise RuntimeError(msg)

#    def cap_gaussians(weights, states, covariances, max_number):
    def cap_gaussians(gaussian_mixture, max_number):
        #sort
        states = gaussian_mixture.mean
        weights = gaussian_mixture.weights
        covariances = gaussian_mixture.covariance
        if len(weights) > max_number:
            indices = np.argsort(weights)
            indices = indices[::-1]
            weights_new = weights()
            weights_new = weights_new * (sum(weights)/sum(weights_new))
            states_new = states[:,indices]
            covariances_new = covariances[:,:,indices]
        else:
            states_new = states
            covariance_new = covariances
            weights_new = weights
    
    def merge_gaussians(weights, states, covariances, merge_threshold):
        number_gaussians = len(weights)
        state_shape = np.shape(states)
        state_dimension = state_shape[0]
    
    def prune_gaussians(weights, states, covariances, prune_threshold):
        number_gaussians = len(weights)
        weights_new = np.array([])
        for ii in range(0, number_gaussians):
            if weights[ii] > prune_threshold:
                np.concatenate(weights_new, weights[ii]) #find a way to do this
                weights_new.append(weights[ii])
                states_new.append
                covariances_new = covariances[:, :]


# %% GMPHD - In progress
class GMPHD:
    def __init__(self):
        # ##TODO: implement
        msg = '{}.{} not implemented'.format(self.__class__.__name__,
                                             self.__init__.__name__)
        raise RuntimeError(msg)


# %% GLMB - In progress
class GLMB:
    def __init__(self):
        # ##TODO: implement
        msg = '{}.{} not implemented'.format(self.__class__.__name__,
                                             self.__init__.__name__)
        raise RuntimeError(msg)