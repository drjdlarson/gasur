# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:57:21 2020

@author: ryan4
"""
import numpy as np
from scipy.stats import multivariate_normal as mvnpdf
from scipy import linalg
from abc import ABC, abstractmethod

from ..enumerations import GuidanceType
from ..exceptions import IncorrectNumberOfTargets


class Guidance(ABC):
    def __init__(self, guidancetype=GuidanceType.NONE, horizon_steps=1,
                 target_states=np.array([])):
        self.type = guidancetype
        self.time_horizon_steps = horizon_steps
        self.target_states = target_states

    @abstractmethod
    def reinitialize_trajectories(self):
        pass

    @abstractmethod
    def update(self):
        pass


class DensityBasedGuidance(Guidance):
    @abstractmethod
    def reinitialize_trajectories(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def density_based_cost(self, obj_weights, obj_states, obj_covariances):
        target_center = self.target_center()
        num_targets = self.target_states.shape[1]
        num_objects = obj_states.shape[1]

        # find radius of influence and shift
        max_dist = 0
        for ii in range(0, num_targets):
            diff = self.target_states[:, ii] - target_center
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
        alpha = 1 / (1 + np.exp(-max_dist - shift))

        # get maximum stand
        max_var_obj = np.max(np.diag(obj_covariances))
        max_var_target = 0
        for ii in range(0, num_targets):
            var = np.max(np.diag(self.target_covariances[:, :, ii].squeeze()))
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
                comb_cov = obj_covariances[:, :, outer_obj].squeeze() \
                            + obj_covariances[:, :, inner_obj].squeeze()
                sum_obj_obj += obj_weights[outer_obj] \
                    * obj_weights[inner_obj] \
                    * mvnpdf.pdf(obj_states[:, outer_obj],
                                 mean=obj_states[:, inner_obj],
                                 covariance=comb_cov)

            # object to target and quadratic
            for tar_ind in range(0, num_targets):
                # object to object
                comb_cov = obj_covariances[:, :, outer_obj].squeeze() \
                            + self.target_covariances[:, :, tar_ind].squeeze()
                sum_obj_target += obj_weights[outer_obj] \
                    * self.target_weights[tar_ind] \
                    * mvnpdf.pdf(obj_states[:, outer_obj],
                                 mean=self.target_states[:, tar_ind],
                                 covariance=comb_cov)

                # quadratic
                diff = obj_states[:, [outer_obj]] \
                    - self.target_states[:, [tar_ind]]
                log_term = np.log((2*np.pi)**(-0.5*num_targets) *
                                  linalg.det(comb_cov)**-0.5) \
                    - 0.5 * diff.transpose() @ linalg.inv(comb_cov) @ diff
                quad += obj_weights[outer_obj] * self.target_weights[tar_ind] \
                    * log_term

        # target to target
        for outer_tar in range(0, num_targets):
            for inner_tar in range(0, num_targets):
                comb_cov = self.target_covariances[:, :, outer_tar].squeeze() \
                            + self.target_covariances[:, :,
                                                      inner_tar].squeeze()
                sum_target_target += self.target_weights[outer_tar] \
                    * self.target_weights[inner_tar] \
                    * mvnpdf.pdf(self.target_states[:, outer_tar],
                                 mean=self.target_states[:, inner_tar],
                                 covariance=comb_cov)

        return 10 * num_objects * max_var_obj * (sum_obj_obj - 20
                                                 * max_var_target * num_targets
                                                 * sum_obj_target) \
            + sum_target_target + alpha*quad

    def update_targets(self, target_states, target_covariances,
                       target_weights):
        self.target_states = target_states
        num_targets = self.target_states.shape[1]

        try:
            given_num = target_covariances.shape[2]
        except IndexError:
            raise IncorrectNumberOfTargets(num_targets,
                                           target_covariances.size)
        if num_targets != given_num:
            raise IncorrectNumberOfTargets(num_targets, given_num)
        else:
            self.target_covariances = target_covariances

        if num_targets != target_weights.size:
            raise IncorrectNumberOfTargets(num_targets, target_weights.size)
        else:
            self.target_weights = target_weights / np.sum(target_weights)

    def target_center(self):
        return np.reshape(np.sum(self.target_states, axis=1)
                          / self.target_states.shape[1],
                          self.targets.shape[0], 1)


# TODO: remove this class and rethink how to structure code here
class LQR(Guidance):
    def __init__(self, guidancetype=GuidanceType.NONE, horizon_steps=1,
                 target_states=np.array([]), Q=np.array([]), R=np.array([])):
        super().__init__(guidancetype=guidancetype,
                         horizon_steps=horizon_steps,
                         target_states=target_states)
        self.state_cost_matrix = Q
        self.control_cost_matrix = R

    def reinitialize_trajectories(self):
        pass

    def update(self):
        pass

    # TODO: change this to basic quadratic cost function, derived classes
    # can implement their own quadratic form
    def cost_function(self, states, control_inputs, desired_state,
                      nominal_control):
        num_time_steps = states.shape[1]
        cost = 0
        for tt in range(0, num_time_steps):
            state_diff = states[:, tt] - desired_state
            control_diff = control_inputs[:, tt] - nominal_control
            cost += state_diff.T @ self.state_cost_matrix  @ state_diff \
                + control_diff.T @ self.control_cost_matrix @ control_diff

    def get_cost_jacobian_hessian(self):
        pass
