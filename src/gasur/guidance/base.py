# -*- coding: utf-8 -*-
"""Implements the base classes for guidance algorithms.

This module contains the classes and data structures used as the base
for guidance related algorithms.
"""
import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn

from gasur.estimator import GaussianMixture
from gasur.utilities.math import get_state_jacobian, get_input_jacobian


class BaseLQR:
    r""" Implements a Linear Quadratic Regulator (LQR) controller.

        This implements an LQR controller for the cost function

        .. math::
            J = \frac{1}{2} \left[x_f^T Q x_f + \int^{t_f}_0 x^T Q x + u^T R u
                     + u^T P x\right]

    Args:
        Q (N x N numpy array): State penalty matrix (default: empty array)
        R (Nu x Nu numpy array): Control penalty matrix (default: empty arary)
        cross_penalty (Nu x N numpy array): Cross penalty matrix (default:
            zero)
        horizon_len (int): Length of trajectory to optimize over (default: Inf)

    Attributes:
        state_penalty (N x N numpy array): State penalty matrix :math:`Q`
        ctrl_penalty (Nu x Nu numpy array): Control penalty matrix :math:`R`
        cross_penalty (Nu x N numpy array): Cross penalty matrix
        horizon_len (int): Length of trajectory to optimize over
    """
    def __init__(self, Q=None, R=None, cross_penalty=None, horizon_len=None,
                 **kwargs):
        if Q is None:
            Q = np.array([[]])
        self.state_penalty = Q
        if R is None:
            R = np.array([[]])
        self.ctrl_penalty = R
        if cross_penalty is None:
            def_rows = self.ctrl_penalty.shape[1]
            if self.state_penalty.size > 0:
                def_cols = self.state_penalty.shape[0]
            else:
                def_cols = 0
            cross_penalty = np.zeros((def_rows, def_cols))
        self.cross_penalty = cross_penalty
        if horizon_len is None:
            horizon_len = np.inf
        self.horizon_len = horizon_len
        super().__init__(**kwargs)

    def iterate(self, F, G, **kwargs):
        """Calculates the feedback gain.

        If using a finite time horizon, this loops over the entire horizon
        to calculate the gain :math:`K` such that the control input is

        .. math::
            u = -Kx

        Args:
            F (N x N numpy array): Discrete time state matrix
            G (N x Nu numpy array): Discrete time input matrix

        Raises:
            RuntimeError: Raised for the finite horizon case

        Todo:
            Implement the finite horizon case

        Returns:
            (Nu x N numpy array): Feedback gain :math:`K`
        """

        if self.horizon_len == np.inf:
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
    """ Implements an Extended Linear Quadratic Regulator (ELQR) controller.

        This implements an ELQR controller for cases where the dynamics are
        non-linear and the cost function is non-quadratic in terms of the
        state. This can be extended to include other forms of non-quadratic
        cost functions.

    Args:
        max_iters (int): Max number of iterations for cost to converge
        horizon_len (int): See :py:class:`gasur.guidance.base.BaseLQR`
            (default: 3), can not be Inf
        cost_tol (float): Tolerance on cost to achieve convergence (default:
            1e-4)

    Raises:
        RuntimeError: If `horizon_len` is Inf

    Attributes:
        max_iters (int): Max number of iterations for cost to converge
        horizon_len (int): See :py:mod:`gasur.guidance.base.baseLQR`
        cost_tol (float): Tolerance on cost to achieve convergence
    """

    def __init__(self, max_iters=50, horizon_len=3, cost_tol=10**-4, **kwargs):
        self.max_iters = max_iters
        self.cost_tol = cost_tol
        super().__init__(horizon_len=horizon_len, **kwargs)
        if self.horizon_len == np.inf:
            raise RuntimeError('Horizon must be finite for ELQR')

    def initialize(self, x_start, n_inputs, **kwargs):
        """ Initialze the start of an iteration.

        Args:
            x_start (N x 1 numpy array): starting state of the iteration
            n_inputs (int): number of control inputs

        Keyword Args:
            feedback (Nu x N numpy array): Feedback matrix
            feedforward (Nu x 1 numpy array): Feedforward matrix
            cost_go_mat (N x N numpy array): Cost-to-go matrix
            cost_go_vec (N x 1 numpy array): Cost-to-go vector
            cost_come_mat (N x N numpy array): Cost-to-come matrix
            cost_come_vec (N x 1 numpy array): Cost-to-come vector

        Returns:
            tuple containing

                - feedback (Nu x N numpy array): Feedback matrix
                - feedforward (Nu x 1 numpy array): Feedforward matrix
                - cost_go_mat (N x N numpy array): Cost-to-go matrix
                - cost_go_vec (N x 1 numpy array): Cost-to-go vector
                - cost_come_mat (N x N numpy array): Cost-to-come matrix
                - cost_come_vec (N x 1 numpy array): Cost-to-come vector
        """
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

    def quadratize_cost(self, x_start, u_nom, timestep, **kwargs):
        r""" Quadratizes the cost function.

        This assumes the true cost function is given

        .. math::
            c_0 &= \frac{1}{2}(x - x_0)^T Q (x - x_0) + \frac{1}{2}(u
                - u_0)^T R (u - u_0) \\
            c_t &= \frac{1}{2}(u - u_{nom})^T R (u - u_{nom})
                + \text{non quadratic state term(s)} \\
            c_{end} &= \frac{1}{2}(x - x_{end})^T Q (x - x_{end})

        Args:
            x_start (N x 1 numpy array): Starting point of the iteration
            u_nom (Nu x 1 numpy array): Nominal control input
            timestep (int): Timestep number, starts at 0 and is relative to
                the begining of the current time horizon
            **kwargs : any arguments needed by
                :py:meth:`gasur.guidance.base.BaseELQR.quadratize_non_quad_state`

        Returns:
            tuple containing

                - P (Nu x N numpy array): cross state penalty matrix
                - Q (N x N numpy array): state penalty matrix
                - R (Nu x Nu numpy array): control input penalty matrix
                - q (N x 1 numpy array): state penalty vector
                - r (Nu x 1 numpy array): control input penalty vector
        """
        if timestep == 0:
            Q = self.state_penalty
            q = -Q @ x_start
        else:
            Q, q = self.quadratize_non_quad_state(**kwargs)

        R = self.ctrl_penalty
        r = -R @ u_nom
        P = np.zeros((u_nom.size, x_start.size))

        return P, Q, R, q, r

    def quadratize_non_quad_state(self, x_hat=None, **kwargs):
        r"""Quadratizes the non-quadratic state.

        This assumes the true cost function is given

        .. math::
            c_0 &= \frac{1}{2}(x - x_0)^T Q (x - x_0) + \frac{1}{2}(u
                - u_0)^T R (u - u_0) \\
            c_t &= \frac{1}{2}(u - u_{nom})^T R (u - u_{nom})
                + \text{non quadratic state term(s)} \\
            c_{end} &= \frac{1}{2}(x - x_{end})^T Q (x - x_{end})

        Args:
            x_hat (N x 1 numpy array): Current state

        Returns:
            tuple containing

                - Q (N x N numpy array): state penalty matrix
                - q (N x 1 numpy array): state penalty vector

        Todo:
            Improve implementation to use the hessian of the non-quadratic
            cost function
        """
        Q = self.state_penalty
        q = np.zeros((x_hat.size, 1))
        return Q, q

    def quadratize_final_cost(self, x_end, **kwargs):
        r"""Quadratizes the cost for the final timestep.

        This assumes the true cost function is given

        .. math::
            c_0 &= \frac{1}{2}(x - x_0)^T Q (x - x_0) + \frac{1}{2}(u
                - u_0)^T R (u - u_0) \\
            c_t &= \frac{1}{2}(u - u_{nom})^T R (u - u_{nom})
                + \text{non quadratic state term(s)} \\
            c_{end} &= \frac{1}{2}(x - x_{end})^T Q (x - x_{end})

        Args:
            x_end (N x 1 numpy array): Desired ending state

        Returns:
            tuple containing

                - Q (N x N numpy array): state penalty matrix
                - q (N x 1 numpy array): state penalty vector
        """
        Q = self.state_penalty
        q = -Q @ x_end

        return Q, q

    def iterate(self, x_start, x_end, u_nom, **kwargs):
        cost_function = kwargs['cost_function']
        dyn_fncs = kwargs['dynamics_fncs']

        feedback, feedforward, cost_go_mat, cost_go_vec, cost_come_mat, \
            cost_come_vec, x_hat = self.initialize(x_start, u_nom.size,
                                                   **kwargs)

        converged = False
        old_cost = 0
        for iteration in range(0, self.max_iters):
            # forward pass
            for kk in range(0, self.horizon_len - 1):
                (x_hat, u_hat, feedback[kk], feedforward[kk],
                 cost_come_mat[kk+1],
                 cost_come_vec[kk+1]) = self.forward_pass(x_hat, feedback[kk],
                                                          feedforward[kk],
                                                          cost_come_mat[kk],
                                                          cost_come_vec[kk],
                                                          cost_go_mat[kk+1],
                                                          cost_go_vec[kk+1],
                                                          kk, x_start=x_start,
                                                          u_nom=u_nom,
                                                          **kwargs)

            # quadratize final cost
            cost_go_mat[-1], cost_go_vec[-1] = \
                self.quadratize_final_cost(x_hat, u_hat, x_end=x_end, **kwargs)
            x_hat = -la.inv(cost_go_mat[-1] + cost_come_mat[-1]) \
                @ (cost_go_vec[-1] + cost_come_vec[-1])

            # backward pass
            for kk in range(self.horizon_len - 2, -1, -1):
                x_hat, u_hat, feedback[kk], feedforward[kk], cost_go_mat[kk], \
                    cost_go_vec[kk] = self.backward_pass(x_hat, feedback[kk],
                                                         feedforward[kk],
                                                         cost_come_mat[kk],
                                                         cost_come_vec[kk],
                                                         cost_go_mat[kk+1],
                                                         cost_go_vec[kk+1],
                                                         kk, x_start=x_start,
                                                         u_nom=u_nom, **kwargs)

            # find real cost of trajectory
            state = x_hat
            cur_cost = 0
            for kk in range(0, self.horizon_len-1):
                ctrl_input = feedback[kk] @ state + feedforward[kk]
                cur_cost += cost_function(state, ctrl_input, **kwargs)
                state = dyn_fncs(state, ctrl_input, **kwargs)
            cur_cost += self.final_cost_function(state, x_end)

            # check for convergence
            if iteration != 0:
                converged = np.abs(old_cost - cur_cost) < self.cost_tol

            old_cost = cur_cost
            if converged:
                break

        return feedback, feedforward

    def final_cost_function(self, state, goal, **kwargs):
        """Cost function at the ending state.

        Args:
            state (numpy array): final state
            goal (numpy array): desired ending state

        Returns:
            (float): cost of the final state

        Raises:
            RuntimeError: if the goal and state are different dimensions
        """
        if state.ndim != 2:
            state = state.reshape((state.size, 1))
        if goal.ndim != 2:
            goal = goal.reshape((goal.size, 1))
        if goal.shape[0] != state.shape[0]:
            msg = 'State ({}) and goal ({}) '.format(state.shape[0],
                                                     goal.shape[0]) \
                + 'do not have the same dimension'.format()
            raise RuntimeError(msg)
        diff = state - goal
        return (diff.T @ self.state_penalty @ diff).squeeze()

    def cost_to_go(self, cost_go_mat, cost_go_vec, P, Q, R, q, r,
                   state_mat, input_mat, c_vec, **kwargs):
        """Calculates the cost-to-go.

        Calculates the cost-to-go matrix and vectors as well as the
        feedforward and feedback gains

        Args:
            cost_go_mat (N x N numpy array): current cost-to-go matrix
            cost_go_vec (N x 1 numpy array): current cost-to-go vector
            P (Nu x N numpy array): cross penalty matrix
            Q (N x N numpy array): state penalty matrix
            R (Nu x Nu numpy array): input penalty matrix
            q (N x 1 numpy array): state penalty vector
            r (Nu x 1 numpy array): input penalty vector
            state_mat (N x N numpy array): state transition matrix
            input_mat (N x Nu numpy array): control input matrix
            c_vec (N x 1): extra vector from the state space equation

        Returns:
            tuple containing

                - cost_go_mat_out (N x N numpy array): cost-to-go matrix
                - cost_go_vec_out (N x 1 numpy array): cost-to-go vector
                - feedback (Nu x N numpy array): feedback gain matrix
                - feedforward (N x 1 numpy array): feedforward gain matrix
        """
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
        """Calculates the cost-to-come.

        Calculates the cost-to-come matrix and vectors as well as the
        feedforward and feedback gains

        Args:
            cost_come_mat (N x N numpy array): current cost-to-go matrix
            cost_come_vec (N x 1 numpy array): current cost-to-go vector
            P (Nu x N numpy array): cross penalty matrix
            Q (N x N numpy array): state penalty matrix
            R (Nu x Nu numpy array): input penalty matrix
            q (N x 1 numpy array): state penalty vector
            r (Nu x 1 numpy array): input penalty vector
            state_mat_bar (N x N numpy array): inverse state transition matrix
            input_mat_bar (N x Nu numpy array): inverse control input matrix
            c_vec_bar (N x 1): extra vector from the inverse state space
                equation

        Returns:
            tuple containing

                - cost_come_mat_out (N x N numpy array): cost-to-go matrix
                - cost_come_vec_out (N x 1 numpy array): cost-to-go vector
                - feedback (Nu x N numpy array): feedback gain matrix
                - feedforward (N x 1 numpy array): feedforward gain matrix
        """
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

    def backward_pass(self, x_hat, u_hat, feedback, feedforward,
                      cost_come_mat, cost_come_vec, cost_go_mat, cost_go_vec,
                      timestep, dyn_fncs, inv_dyn_fncs, **kwargs):
        """ Implements the backward pass of the ELQR algorithm for a single
        timestep.

        Args:
            x_hat (N x 1 numpy array): current state
            u_hat (N x 1 numpy array): current input
            feedback (Nu x N numpy array): feedback gain matrix
            feedforward (Nu x 1 numpy array): feedforward gain matrix
            cost_come_mat (N x N numpy array): cost-to-come matrix
            cost_come_vec (N x 1 numpy array): cost-to-come vector
            cost_go_mat (N x N numpy array): cost-to-go matrix
            cost_go_vec (N x 1 numpy array): cost-to-go vector
            timestep (int): current time step, starts at 0 every time horizon
            dyn_fncs (list of functions): dynamics functions, one per state,
                must take in x, u as parameters
            inv_dyn_fncs (list of functions): inverse dynamics functions,
                one per state, must take in x, u as parameters
            **kwargs : passed through to
                :py:meth:`gasur.guidance.base.BaseELQR.quadratize_cost`

        Returns:
            tuple containing

                - x_hat (N x 1 numpy array): prior state
                - feedback (Nu x N numpy array): prior feedback gain
                - feedforward (N x 1 numpy array): prior feedforward gain
                - cost_go_mat_out (N x N numpy array): prior cost-to-go matrix
                - cost_go_vec_out (N x 1 numpy array): prior cost-to-go vector
        """
        x_hat_prime = np.zeros(x_hat.shape)
        for ii, gg in enumerate(inv_dyn_fncs):
            x_hat_prime[ii] = gg(x_hat, u_hat, **kwargs)

        state_mat = get_state_jacobian(x_hat_prime, u_hat,
                                       dyn_fncs, **kwargs)
        input_mat = get_input_jacobian(x_hat_prime, u_hat,
                                       dyn_fncs, **kwargs)
        c_vec = x_hat - state_mat @ x_hat_prime - input_mat @ u_hat

        P, Q, R, q, r = self.quadratize_cost(x_hat=x_hat_prime,
                                             timestep=timestep, **kwargs)

        (cost_go_mat_out, cost_go_vec_out, feedback,
         feedforward) = self.cost_to_go(cost_go_mat, cost_go_vec, P, Q, R, q,
                                        r, state_mat, input_mat, c_vec)

        # update state estimate
        x_hat = -la.inv(cost_go_mat_out + cost_come_mat) \
            @ (cost_go_vec_out + cost_come_vec)

        return (x_hat, feedback, feedforward, cost_go_mat_out,
                cost_go_vec_out)

    def forward_pass(self, x_hat, u_hat, feedback, feedforward, cost_come_mat,
                     cost_come_vec, cost_go_mat, cost_go_vec, timestep,
                     dyn_fncs, inv_dyn_fncs, **kwargs):
        """ Implements the forward pass of the ELQR algorithm for a single
        timestep.

        Args:
            x_hat (N x 1 numpy array): current state
            u_hat (N x 1 numpy array): current input
            feedback (Nu x N numpy array): feedback gain matrix
            feedforward (Nu x 1 numpy array): feedforward gain matrix
            cost_come_mat (N x N numpy array): cost-to-come matrix
            cost_come_vec (N x 1 numpy array): cost-to-come vector
            cost_go_mat (N x N numpy array): cost-to-go matrix
            cost_go_vec (N x 1 numpy array): cost-to-go vector
            timestep (int): current time step, starts at 0 every time horizon
            dyn_fncs (list of functions): dynamics functions, one per state,
                must take in x, u as parameters
            inv_dyn_fncs (list of functions): inverse dynamics functions,
                one per state, must take in x, u as parameters
            **kwargs : passed through to
                :py:meth:`gasur.guidance.base.BaseELQR.quadratize_cost`

        Returns:
            tuple containing

                - x_hat (N x 1 numpy array): next state
                - feedback (Nu x N numpy array): next feedback gain
                - feedforward (N x 1 numpy array): next feedforward gain
                - cost_go_mat_out (N x N numpy array): next cost-to-go matrix
                - cost_go_vec_out (N x 1 numpy array): next cost-to-go vector
        """
        x_hat_prime = np.zeros(x_hat.shape)
        for ii, ff in enumerate(dyn_fncs):
            x_hat_prime[ii] = ff(x_hat, u_hat, **kwargs)

        state_mat_bar = get_state_jacobian(x_hat_prime, u_hat,
                                           inv_dyn_fncs, **kwargs)
        input_mat_bar = get_input_jacobian(x_hat_prime, u_hat,
                                           inv_dyn_fncs, **kwargs)
        c_bar_vec = x_hat - state_mat_bar @ x_hat_prime \
            - input_mat_bar @ u_hat

        P, Q, R, q, r = self.quadratize_cost(x_hat=x_hat, timestep=timestep,
                                             **kwargs)

        (cost_come_mat_out, cost_come_vec_out, feedback,
         feedforward) = self.cost_to_come(cost_come_mat, cost_come_vec,
                                          P, Q, R, q, r, state_mat_bar,
                                          input_mat_bar, c_bar_vec)

        # update state estimate
        x_hat = -la.inv(cost_go_mat + cost_come_mat_out) \
            @ (cost_go_vec + cost_come_vec_out)

        return (x_hat, feedback, feedforward, cost_come_mat_out,
                cost_come_vec_out)


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
        self.weight = kwargs.get('weight', 0)
        if self.weight <= 0:
            raise ValueError('Weight must be greater than 0')
        self.ctrl_nom = kwargs.get('ctrl_nom', np.array([[]]))
