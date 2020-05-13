"""Implements RFS guidance algorithms.

This module contains the classes and data structures
for RFS guidance related algorithms.
"""
import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn
from scipy.stats.distributions import chi2

from gasur.estimator import GaussianMixture
from gncpy.math import get_hessian, get_jacobian
from gncpy.control import BaseELQR


class DensityBased:
    """Defines the base data structure for guidance using Gaussian Mixtures.

    Args:
        wayareas (GaussianMixture): desired target distributions, see
            :py:class:`gasur.estimator.GaussianMixture`
        safety_factor (float): overbounding saftey factor when calculating the
            radius of influence for the activation function (default: 1).
        y_ref (float): Reference point on the sigmoid, must be less than 1
            (default: 0.9)

    Raises:
        ValueError: if y_ref is greater than 1

    Attributes:
        targets (GaussianMixture): desired target distributions, see
            :py:class:`gasur.estimator.GaussianMixture`
        safety_factor (float): overbounding saftey factor when calculating the
            radius of influence for the activation function.
        y_ref (float): Reference point on the sigmoid, must be less than 1
    """
    def __init__(self, wayareas=None, safety_factor=1, y_ref=0.9, **kwargs):
        if wayareas is None:
            wayareas = GaussianMixture()
        self.targets = wayareas
        self.safety_factor = safety_factor
        if y_ref >= 1:
            raise ValueError('Reference point must be less than 1')
        self.y_ref = y_ref
        super().__init__(**kwargs)

    def density_based_cost(self, obj_states, obj_weights, obj_covariances,
                           **kwargs):
        r"""Implements the density based cost function.

        Implements the following cost function based on the difference between
        Gaussian mixtures, with additional terms to improve convergence when
        far from the targets.

        .. math::
            J &= \sum_{k=1}^{T} 10 N_g\sigma_{g,max}^2 \left( \sum_{j=1}^{N_g}
                    \sum_{i=1}^{N_g} w_{g,k}^{(j)} w_{g,k}^{(i)}
                    \mathcal{N}( \mathbf{m}^{(j)}_{g,k}; \mathbf{m}^{(i)}_{g,k},
                    P^{(j)}_{g, k} + P^{(i)}_{g, k} ) \right. \\
                &- \left. 20 \sigma_{d, max}^2 N_d \sum_{j=1}^{N_d} \sum_{i=1}^{N_g}
                    w_{d,k}^{(j)} w_{g,k}^{(i)} \mathcal{N}(
                    \mathbf{m}^{(j)}_{d, k}; \mathbf{m}^{(i)}_{g, k},
                    P^{(j)}_{d, k} + P^{(i)}_{g, k} ) \right) \\
                &+ \sum_{j=1}^{N_d} \sum_{i=1}^{N_d} w_{d,k}^{(j)}
                    w_{d,k}^{(i)} \mathcal{N}( \mathbf{m}^{(j)}_{d,k};
                    \mathbf{m}^{(i)}_{d,k}, P^{(j)}_{d, k} + P^{(i)}_{d, k} ) \\
                &+ \alpha \sum_{j=1}^{N_d} \sum_{i=1}^{N_g} w_{d,k}^{(j)}
                    w_{g,k}^{(i)} \ln{\mathcal{N}( \mathbf{m}^{(j)}_{d,k};
                    \mathbf{m}^{(i)}_{g,k}, P^{(j)}_{d, k} + P^{(i)}_{g, k} )}

        Args:
            obj_states (N x Ng numpy array): Matrix of all the object's states,
                each column is one objects state
            obj_weights (list of floats): weight of each state, same order as
                obj_states
            obj_covariances (list): list of N x N numpy arrays representing
                each states covariance matrix

        Returns:
            (float): density based cost
        """
        target_center = self.target_center()
        num_targets = len(self.targets.means)
        num_objects = obj_states.shape[1]
        num_states = obj_states.shape[0]

        # find radius of influence and shift
        max_dist = 0
        for tar_mean in self.targets.means:
            diff = tar_mean - target_center
            dist = np.sqrt(diff.T @ diff).squeeze()
            if dist > max_dist:
                max_dist = dist
        radius_of_influence = self.safety_factor * max_dist
        shift = radius_of_influence + np.log(1/self.y_ref - 1)

        # get actiavation term
        max_dist = 0
        for ii in range(0, num_objects):
            diff = target_center - obj_states[:, [ii]]
            dist = np.sqrt(diff.T @ diff).squeeze()
            if dist > max_dist:
                max_dist = dist
        activator = 1 / (1 + np.exp(-(max_dist - shift)))

        # get maximum variance
        max_var_obj = max(map(lambda x: float(np.max(np.diag(x))),
                              obj_covariances))
        max_var_target = max(map(lambda x: float(np.max(np.diag(x))),
                                 self.targets.covariances))

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
                    * mvn.pdf(obj_states[:, outer_obj],
                              mean=obj_states[:, inner_obj],
                              cov=comb_cov)

            # object to target and quadratic
            for ii in range(0, num_targets):
                # object to target
                comb_cov = obj_covariances[outer_obj] \
                    + self.targets.covariances[ii]
                sum_obj_target += obj_weights[outer_obj] \
                    * self.targets.weights[ii] \
                    * mvn.pdf(obj_states[:, outer_obj],
                              mean=self.targets.means[ii].squeeze(),
                              cov=comb_cov)

                # quadratic
                diff = obj_states[:, [outer_obj]] - self.targets.means[ii]
                log_term = np.log((2*np.pi)**(-0.5*num_states)
                                  / np.sqrt(la.det(comb_cov))) \
                    - 0.5 * diff.T @ la.inv(comb_cov) @ diff
                quad += (obj_weights[outer_obj] * self.targets.weights[ii]
                         * log_term).squeeze()

        # target to target
        for outer in range(0, num_targets):
            for inner in range(0, num_targets):
                comb_cov = self.targets.covariances[outer] \
                    + self.targets.covariances[inner]
                sum_target_target += self.targets.weights[outer] \
                    * self.targets.weights[inner] \
                    * mvn.pdf(self.targets.means[outer].squeeze(),
                              mean=self.targets.means[inner].squeeze(),
                              cov=comb_cov)

        return 10 * num_objects * max_var_obj * (sum_obj_obj - 2
                                                 * max_var_target * num_targets
                                                 * sum_obj_target) \
            + sum_target_target + activator * quad

    def convert_waypoints(self, waypoints):
        """Converts waypoints into wayareas.

        Args:
            waypoints (list): each element is a N x 1 numpy array representing
                a desired state

        Returns:
            (GaussianMixture): desired wayareas, see :py:class:`gasur.estimator.GaussianMixture`

        Todo:
            Ensure the calculated covariance is full rank and positive
            semi-definite
        """
        center_len = waypoints[0].size
        num_waypoints = len(waypoints)
        combined_centers = np.zeros((center_len, num_waypoints+1))

        # get overall center, build collection of centers
        for ii in range(0, num_waypoints):
            combined_centers[:, [ii]] = waypoints[ii].reshape((center_len, 1))
            combined_centers[:, [num_waypoints]] += combined_centers[:, [ii]]
        combined_centers[:, [num_waypoints]] = combined_centers[:, [-1]] \
            / num_waypoints

        # find directions to each point
        directions = np.zeros((center_len, num_waypoints + 1, num_waypoints
                               + 1))
        for start_point in range(0, num_waypoints + 1):
            for end_point in range(0, num_waypoints + 1):
                directions[:, start_point, end_point] = \
                    (combined_centers[:, end_point]
                     - combined_centers[:, start_point])

        def find_principal_components(data):
            num_samps = data.shape[0]
            num_feats = data.shape[1]

            mean = np.sum(data, 0) / num_samps
            covars = np.zeros((num_feats, num_feats))
            for ii in range(0, num_feats):
                for jj in range(0, num_feats):
                    acc = 0
                    for samp in range(0, num_samps):
                        acc += (data[samp, ii] - mean[ii]) \
                                * (data[samp, jj] - mean[jj])
                    covars[ii, jj] = acc / num_samps
            (w, comps) = la.eig(covars)
            inds = np.argsort(w)[::-1]
            return comps[:, inds].T

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

            sample_data = combined_centers.copy()
            sample_data = np.delete(sample_data, wp_ind, 1)
            sample_dirs = directions[:, wp_ind, :].squeeze()
            sample_dirs = np.delete(sample_dirs, wp_ind, 1)
            comps = find_principal_components(sample_data.T)
            vals = find_largest_proj_dist(comps, sample_dirs)

            covariance = comps @ vals @ la.inv(comps)

            wayareas.means.append(center)
            wayareas.covariances.append(covariance)
            wayareas.weights.append(weight)
        return wayareas

    def update_targets(self, new_waypoints, **kwargs):
        """Updates the target list.

        Args:
            new_waypoints (list): each element is a N x 1 numpy array
                representing at target state

        Keyword Args:
            reset (bool): Removes the current targets if true, else appends
                new_waypoints to the current set of targets
        """
        reset = kwargs.get('reset', False)
        if not reset:
            for m in self.targets.means:
                new_waypoints.append(m)
        # clear way areas and add new ones
        self.targets = GaussianMixture()
        self.targets = self.convert_waypoints(new_waypoints)

    def target_center(self):
        """Calculates the mean of the overall target distribution.

        Returns:
            (N x 1 numpy array): the mean of the target states
        """
        summed = np.zeros(self.targets.means[0].shape)
        num_tars = len(self.targets.means)
        for ii in range(0, num_tars):
            summed += self.targets.means[ii]
        return summed / num_tars


class GaussianObject:
    """Data structure for an object defined by a Gaussian distribution.

    This defines the attributes for a general object used in
    Gaussian based guidance algorithms.

    Keyword Args:
        dyn_functions (list of functions): list of dynamics functions, 1 per
            state, same order as state variables, must take in x, u
        inv_dyn_functions (list of functions): list of inverse dynamics
            functions, 1 per state, same order as state variables, must take
            in x, u
        means (Nh x N numpy array): the state at each timestep of the time
            horizon
        ctrl_inputs (Nh x Nu numpy array): control input at each timestep of
            the time horizon
        feedforward (list): list of numpy arrays of the Nu x 1 feedforward
            gain, one for each timestep of the time horizon
        feedback (list): list of numpy arrays of the Nu x N feedforward
            gain, one for each timestep of the time horizon
        cost_to_come_mat (list): list of numpy arras of the N x N cost-to-come
            matrix, one for wach timestep of the time horizon
        cost_to_come_vec (list): list of numpy arras of the N x 1 cost-to-come
            vector, one for wach timestep of the time horizon
        cost_to_go_mat (list): list of numpy arras of the N x N cost-to-go
            matrix, one for wach timestep of the time horizon
        cost_to_go_mat (list): list of numpy arras of the N x 1 cost-to-go
            vector, one for wach timestep of the time horizon
        covariance (N x N numpy array): covariance matrix
        weight (float): weight of the Gaussian in the mixture, must be greater
            than 0
        ctrl_nom (Nu x 1 numpy array): nominal control input

    Raises:
        ValueError: if weight is less than or equal to 0

    Attributes:
        dyn_functions (list of functions): list of dynamics functions, 1 per
            state, same order as state variables, must take in x, u
        inv_dyn_functions (list of functions): list of inverse dynamics
            functions, 1 per state, same order as state variables, must take
            in x, u
        means (Nh x N numpy array): the state at each timestep of the time
            horizon
        ctrl_inputs (Nh x Nu numpy array): control input at each timestep of
            the time horizon
        feedforward (list): list of numpy arrays of the Nu x 1 feedforward
            gain, one for each timestep of the time horizon
        feedback (list): list of numpy arrays of the Nu x N feedforward
            gain, one for each timestep of the time horizon
        cost_to_come_mat (list): list of numpy arras of the N x N cost-to-come
            matrix, one for wach timestep of the time horizon
        cost_to_come_vec (list): list of numpy arras of the N x 1 cost-to-come
            vector, one for wach timestep of the time horizon
        cost_to_go_mat (list): list of numpy arras of the N x N cost-to-go
            matrix, one for wach timestep of the time horizon
        cost_to_go_vec (list): list of numpy arras of the N x 1 cost-to-go
            vector, one for wach timestep of the time horizon
        covariance (N x N numpy array): covariance matrix, only 1 for entire
            time horizon
        weight (float): weight of the Gaussian in the mixture, must be greater
            than 0
        ctrl_nom (Nu x 1 numpy array): nominal control input
    """

    def __init__(self, **kwargs):
        self.dyn_functions = kwargs.get('dyn_functions', [])
        self.inv_dyn_functions = kwargs.get('inv_dyn_functions', [])

        # each timestep is a row
        self.means = kwargs.get('means', np.array([[]]))
        self.ctrl_inputs = kwargs.get('control_input', np.array([[]]))

        # lists of arrays
        self.feedforward_lst = kwargs.get('feedforward', [])
        self.feedback_lst = kwargs.get('feedback', [])
        self.cost_to_come_mat = kwargs.get('cost_to_come_mat', [])
        self.cost_to_come_vec = kwargs.get('cost_to_come_vec', [])
        self.cost_to_go_mat = kwargs.get('cost_to_go_mat', [])
        self.cost_to_go_vec = kwargs.get('cost_go_vec', [])

        # only 1 for entire trajectory
        self.covariance = kwargs.get('covariance', np.array([[]]))
        self.weight = kwargs.get('weight', np.nan)
        if self.weight <= 0:
            raise ValueError('Weight must be greater than 0')
        self.ctrl_nom = kwargs.get('ctrl_nom', np.array([[]]))



class ELQRGaussian(BaseELQR, DensityBased):
    r""" Implements centralized Extended LQR for gaussian mixtures.

    Args:
        cur_gaussians (list): list of gaussian objects
        similar_thresh (float): minimum probability threshold for
            :math:`\chi^2` similarity test

    Raises:
        VaueError: if similar_thresh is greater than or equal to 1

    Attributes:
        cur_gaussians (list): list of gaussian objects
        similar_thresh (float): minimum probability threshold for
            :math:`\chi^2` similarity test
    """

    def __init__(self, cur_gaussians=None, similar_thresh=0.95, **kwargs):
        if cur_gaussians is None:
            cur_gaussians = []
        self.gaussians = cur_gaussians  # list of GaussianObjects
        self.similar_thresh = similar_thresh
        if self.similar_thresh >= 1:
            raise ValueError('similar_thresh must be less than 1')
        super().__init__(**kwargs)

    def initialize(self, measured_gaussians, est_dyn_lst, est_inv_dyn_lst,
                   n_inputs_lst, **kwargs):
        """ (Re-)initializes for current optimization attempt.

        Converts measured values into class data structures and compares with
        values calculated from the last call
        to :py:meth:`gasur.guidance.centralized.ELQRGaussian.iterate`. If
        none match, then the measurents are held and remaining properties set
        to zero. Overrides base class version.

        Args:
            meausred_gaussians (GaussianMixture): currently observed Gaussians
            est_dyn_lst (list): each element is a list of dynamics functions,
                must take x, u as parameters
            est_inv_dyn_lst (list): each element is a list of inverse dynamics
                functions, must take x, u as parameters
            n_inputs_lst (list): each element is the number of control inputs
                for the corresponding dynamics functions

        Keyword Args:
            u_nom_lst (list): list of Nu x 1 numpy arrays representing the
                nominal control input for the corresponding dynamics functions
        """
        # Inputs: GaussianMixture object of observations
        u_nom_lst = kwargs.pop('ctrl_nom_lst', None)

        num_obs = len(measured_gaussians.means)
        if num_obs == 0:
            self.gaussians = []
            return

        n_states = measured_gaussians.means[0].size
        new_gaussians = []
        for ii in range(0, num_obs):
            obs_mean = measured_gaussians.means[ii].reshape((n_states, 1))
            best_fit = (-1, np.inf)  # index, fit criteria

            for jj, gg in enumerate(self.gaussians):
                test_mean = gg.means[[1], :].reshape((n_states, 1))
                test_cov = gg.covariance

                diff = obs_mean - test_mean
                fit_criteria = diff.T @ la.inv(test_cov) @ diff
                if fit_criteria < best_fit[1]:
                    best_fit = (jj, fit_criteria)

            found_match = best_fit[1] < chi2.ppf(self.similar_thresh,
                                                 df=n_states)

            # initialize GaussianObject with measured value ii
            obj = GaussianObject()
            obj.means = np.zeros((self.horizon_len, n_states))
            obj.means[0, :] = measured_gaussians.means[ii].squeeze()
            obj.covariance = measured_gaussians.covariances[ii]
            obj.weight = measured_gaussians.weights[ii]
            if found_match:
                # Init remaining trajectory of ii with self.gaussian jj
                ind = best_fit[0]
                obj.dyn_functions = self.gaussians[ind].dyn_functions
                obj.inv_dyn_functions = self.gaussians[ind].inv_dyn_functions
                obj.ctrl_nom = self.gaussians[ind].ctrl_nom

                n_inputs = self.gaussians[0].ctrl_inputs.shape[1]
                obj.ctrl_inputs = np.zeros((self.horizon_len, n_inputs))
                obj.ctrl_inputs[:-1, :] = self.gaussians[ind].ctrl_inputs[1::, :]

                obj.means[1:-1, :] = self.gaussians[ind].means[2::, :]
                for jj, ff in enumerate(obj.dyn_functions):
                    obj.means[-1, jj] = ff(obj.means[[-2], :].T,
                                           obj.ctrl_inputs[[-2], :].T,
                                           **kwargs)
                obj.feedforward_lst = self.gaussians[ind].feedforward_lst[1::]
                shape = self.gaussians[ind].feedforward_lst[0].shape
                obj.feedforward_lst.append(np.zeros(shape))

                obj.feedback_lst = self.gaussians[ind].feedback_lst[1::]
                shape = self.gaussians[ind].feedforward_lst[0].shape
                obj.feedback_lst.append(np.zeros(shape))

                obj.cost_to_come_mat = self.gaussians[ind].cost_to_come_mat[1::]
                obj.cost_to_come_mat.append(obj.cost_to_come_mat[-1].copy())
                obj.cost_to_come_vec = self.gaussians[ind].cost_to_go_vec[1::]
                obj.cost_to_come_vec.append(obj.cost_to_come_vec[-1].copy())
                obj.cost_to_go_mat = self.gaussians[ind].cost_to_go_mat[1::]
                obj.cost_to_go_mat.append(obj.cost_to_go_mat[-1].copy())
                obj.cost_to_go_vec = self.gaussians[ind].cost_to_go_vec[1::]
                obj.cost_to_go_vec.append(obj.cost_to_go_vec[-1].copy())

                # remove self.gaussian jj
                self.gaussians.pop(ind)
            else:
                # Predict remaining time horizon
                obj.means[1::, :] = obj.means[0, :]
                obj.ctrl_input = np.zeros((self.horizon_len, n_inputs_lst[ii]))
                obj.dyn_functions = est_dyn_lst[ii].copy()
                obj.inv_dyn_functions = est_inv_dyn_lst[ii].copy()
                if u_nom_lst is None:
                    u_nom = np.zeros((n_inputs_lst[ii], 1))
                else:
                    u_nom = u_nom_lst[ii]
                obj.ctrl_nom = u_nom
                ff = np.zeros((u_nom.shape[0], 1))
                fb = np.zeros((u_nom.shape[0], n_states))
                ccm = np.zeros((n_states, n_states))
                ccv = np.zeros((n_states, 1))
                cgm = np.zeros((n_states, n_states))
                cgv = np.zeros((n_states, 1))
                for jj in range(0, self.horizon_len):
                    obj.feedforward_lst.append(ff.copy())
                    obj.feedback_lst.append(fb.copy())
                    obj.cost_to_come_mat.append(ccm.copy())
                    obj.cost_to_come_vec.append(ccv.copy())
                    obj.cost_to_go_mat.append(cgm.copy())
                    obj.cost_to_go_vec.append(cgv.copy())

            # add newly initialized object to list
            new_gaussians.append(obj)
        # assign new gaussians to class variable
        self.gaussians = new_gaussians

    def quadratize_non_quad_state(self, all_states=None, obj_num=None,
                                  **kwargs):
        """Quadratizes the non-quadratic state terms in the cost function.

        Overrides the base class version,
        see :py:meth:`gasur.guidance.base.BaseELQR.quadratize_non_quad_state`

        Args:
            all_states (N x Ng numpy array): matrix containing states of all
                gaussians for current timestep
            obj_num (int): index of the guassian object currently being
                evaluated
            **kwargs : passed through
                to :py:meth:`gasur.utilities.math.get_hessian`

        Returns:
            tuple containing

                - Q (N x N numpy array): state penalty matrix
                - q (N x 1 numpy array): state penalty vector
        """
        def helper(x, cur_states):
            loc_states = cur_states.copy()
            loc_states[:, [obj_num]] = x.copy()
            weight_lst = []
            cov_lst = []
            for ii in self.gaussians:
                weight_lst.append(ii.weight)
                cov_lst.append(ii.covariance)
            return self.density_based_cost(loc_states, weight_lst, cov_lst)

        Q = get_hessian(all_states[:, [obj_num]].copy(),
                        lambda x_: helper(x_, all_states), **kwargs)

        # Regularize Matrix
        eig_vals, eig_vecs = la.eig(Q)
        for ii in range(0, eig_vals.size):
            if eig_vals[ii] < 0:
                eig_vals[ii] = 0
        Q = eig_vecs @ np.diag(eig_vals) @ la.inv(eig_vecs)

        q = get_jacobian(all_states[:, [obj_num]].copy(),
                         lambda x_: helper(x_, all_states), **kwargs)
        q = q - Q @ all_states[:, [obj_num]]

        return Q, q

    def iterate(self, measured_gaussians, **kwargs):
        """Performs one iteration of the ELQR over the entire time horizon.

        Overrides base class,
        see :py:meth:`gasur.guidance.base.BaseELQR.iterate`

        Args:
            measured_gaussians (GaussianMixture): currently measured gaussians

        Keyword Args:
            est_dyn_lst (list): each element is a list of dynamics functions,
                must take x, u as parameters
            est_inv_dyn_lst (list): each element is a list of inverse dynamics
                functions, must take x, u as parameters
            n_inputs_lst (list): each element is the number of control inputs
                for the corresponding dynamics functions
        """
        est_dyn_lst = kwargs.pop('est_dyn_lst')
        est_inv_dyn_lst = kwargs.pop('est_inv_dyn_lst')
        n_inputs_lst = kwargs.pop('n_inputs_lst')

        self.initialize(measured_gaussians, est_dyn_lst, est_inv_dyn_lst,
                        n_inputs_lst, **kwargs)
        num_gaussians = len(self.gaussians)
        if num_gaussians == 0:
            return

        x_starts = np.zeros((num_gaussians, self.gaussians[0].means.shape[1]))
        for ii, gg in enumerate(self.gaussians):
            x_starts[ii, :] = gg.means[1, :]

        converged = False
        old_cost = np.inf
        for iteration in range(0, self.max_iters):
            # forward pass
            for kk in range(0, self.horizon_len - 1):
                cur_states = np.zeros((num_gaussians,
                                       self.gaussians[0].means.shape[1]))
                for ii, gg in enumerate(self.gaussians):
                    cur_states[ii, :] = gg.means[kk, :]

                # update control for each gaussian
                for gg in self.gaussians:
                    gg.ctrl_input[kk, :] = (gg.feedback_lst[kk]
                                            @ gg.means[[kk], :].T
                                            + gg.feedforward_lst[kk]).squeeze()

                for ii, gg in enumerate(self.gaussians):
                    x_hat = gg.means[[kk], :].T
                    u_hat = gg.ctrl_input[[kk], :].T
                    feedback = gg.feedback_lst[kk]
                    feedforward = gg.feedforward_lst[kk]
                    cost_come_mat = gg.cost_to_come_mat[kk]
                    cost_come_vec = gg.cost_to_come_vec[kk]
                    cost_go_mat = gg.cost_to_go_mat[kk+1]
                    cost_go_vec = gg.cost_to_go_vec[kk+1]
                    x_start = x_starts[[ii], :].T
                    u_nom = gg.ctrl_nom
                    f = gg.dyn_functions
                    in_f = gg.inv_dyn_functions

                    (x_hat, gg.feedback_lst[kk], gg.feedforward_lst[kk],
                     gg.cost_to_come_mat[kk+1],
                     gg.cost_to_come_vec[kk+1]) = self.forward_pass(x_hat,
                                                                    u_hat,
                                                                    feedback,
                                                                    feedforward,
                                                                    cost_come_mat,
                                                                    cost_come_vec,
                                                                    cost_go_mat,
                                                                    cost_go_vec,
                                                                    kk,
                                                                    x_start=x_start,
                                                                    u_nom=u_nom,
                                                                    dyn_fncs=f,
                                                                    inv_dyn_fncs=in_f,
                                                                    all_states=cur_states.T,
                                                                    obj_num=ii,
                                                                    **kwargs)
                    gg.means[[kk+1], :] = x_hat.T

            # quadratize final cost
            for gg in self.gaussians:
                x_hat = gg.means[[-1], :].T
                u_hat = gg.ctrl_input[[-1], :].T
                x_end = self.find_nearest_target(gg.means[-1, :])
                gg.cost_to_go_mat[-1], gg.cost_to_go_vec[-1] = \
                    self.quadratize_final_cost(x_hat, u_hat, x_end=x_end,
                                               **kwargs)
                gg.means[-1, :] = (-la.inv(gg.cost_to_go_mat[-1]
                                          + gg.cost_to_come_mat[-1])
                                   @ (gg.cost_to_go_vec[-1]
                                      + gg.cost_to_come_vec[-1])).squeeze()

            # backward pass
            for kk in range(self.horizon_len - 2, -1, -1):
                prev_states = np.zeros((num_gaussians,
                                       self.gaussians[0].means.shape[1]))
                for ii, gg in enumerate(self.gaussians):
                    gg.ctrl_input[kk, :] = (gg.feedback_lst[kk]
                                            @ gg.means[[kk+1], :].T
                                            + gg.feedforward_lst[kk]).squeeze()
                    for jj, ff in enumerate(gg.inv_dyn_functions):
                        prev_states[ii, jj] = ff(gg.means[kk+1, :],
                                                 gg.ctrl_input[kk, :],
                                                 **kwargs)

                # update values
                for ii, gg in enumerate(self.gaussians):
                    x_hat = gg.means[[kk+1], :].T
                    u_hat = gg.ctrl_input[[kk], :].T
                    feedback = gg.feedback_lst[kk]
                    feedforward = gg.feedforward_lst[kk]
                    cost_come_mat = gg.cost_to_come_mat[kk]
                    cost_come_vec = gg.cost_to_come_vec[kk]
                    cost_go_mat = gg.cost_to_go_mat[kk+1]
                    cost_go_vec = gg.cost_to_go_vec[kk+1]
                    x_start = x_starts[[ii], :].T
                    u_nom = gg.ctrl_nom
                    f = gg.dyn_functions
                    in_f = gg.inv_dyn_functions

                    (x_hat, gg.feedback_lst[kk], gg.feedforward_lst[kk],
                     gg.cost_to_go_mat[kk],
                     gg.cost_to_go_vec[kk]) = self.backward_pass(x_hat, u_hat,
                                                                 feedback,
                                                                 feedforward,
                                                                 cost_come_mat,
                                                                 cost_come_vec,
                                                                 cost_go_mat,
                                                                 cost_go_vec,
                                                                 kk,
                                                                 x_start=x_start,
                                                                 u_nom=u_nom,
                                                                 dyn_fncs=f,
                                                                 inv_dyn_fncs=in_f,
                                                                 all_states=prev_states.T,
                                                                 obj_num=ii,
                                                                 **kwargs)
                    gg.means[[kk], :] = x_hat.T

            # find real cost of trajectory
            states = x_starts
            weight_lst = []
            cov_lst = []
            n_inputs = self.gaussians[0].feedback_lst[kk].shape[0]
            for gg in self.gaussians:
                weight_lst.append(gg.weight)
                cov_lst.append(gg.covariance)
            cur_cost = 0
            for kk in range(0, self.horizon_len-1):
                cur_cost += self.density_based_cost(states.T, weight_lst,
                                                    cov_lst, **kwargs)

                ctrl_inputs = np.zeros((num_gaussians, n_inputs))
                for ii, gg in enumerate(self.gaussians):
                    state = states[[ii], :].T
                    ctrl_inputs[ii, :] = (gg.feedback_lst[kk] @ state
                                          + gg.feedforward_lst[kk]).squeeze()
                    for jj, ff in enumerate(gg.dyn_functions):
                        states[ii, jj] = ff(state, ctrl_inputs[[ii], :].T,
                                            **kwargs)
            cur_cost += self.final_cost_function(states)

            # check for convergence
            converged = np.abs(old_cost - cur_cost) < self.cost_tol

            old_cost = cur_cost
            if converged:
                break

    def final_cost_function(self, all_states, **kwargs):
        """Calculates the true cost at the final timestep.
        
        Wrapper around base class version,
        see :py:meth:`gasur.guidance.base.BaseELQR.final_cost_function`

        Args:
            all_states (Ng x N numpy array): array of the ending statess for
                all gaussians
            **kwargs : passed through to base class version
        """
        cost = 0
        for state in all_states:
            goal = self.find_nearest_target(state)
            cost += super().final_cost_function(state, goal, **kwargs)
        return cost

    def find_nearest_target(self, state):
        """ Finds target closest to the given state.

        Uses the :math:`L_2` distance to find the closest target.

        Args:
            state (N x 1 numpy array): current state

        Returns:
            (N x 1 numpy array): nearest target
        """
        x_end = []
        min_dist = np.inf
        for goal in self.targets.means:
            dist = la.norm(state.squeeze() - goal.squeeze())
            if dist < min_dist:
                min_dist = dist
                x_end = goal.copy()
        return x_end
