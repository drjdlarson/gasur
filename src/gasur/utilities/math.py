# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:22:52 2020

@author: ryan4
"""
import numpy as np


def_step_size = 10**-7


def get_jacobian(x, fnc, **kwargs):
    step_size = kwargs.get('step_size', def_step_size)
    inv_step2 = 1 / (2 * step_size)
    n_vars = x.size
    J = np.zeros((n_vars, 1))
    for ii in range(0, n_vars):
        x_r = x.copy()
        x_l = x.copy()
        x_r[ii] += step_size
        x_l[ii] -= step_size
        J[ii] = inv_step2 * (fnc(x_r) - fnc(x_l))
    return J


def get_state_jacobian(x, u, fncs, **kwargs):
    n_states = x.size
    A = np.zeros((n_states, n_states))
    for row in range(0, n_states):
        A[[row], :] = get_jacobian(x.copy(),
                                   lambda x_: fncs[row](x_, u, **kwargs),
                                   **kwargs).T
    return A


def get_input_jacobian(x, u, fncs, **kwargs):
    n_states = x.size
    n_inputs = u.size
    B = np.zeros((n_states, n_inputs))
    for row in range(0, n_states):
        B[[row], :] = get_jacobian(u.copy(),
                                   lambda u_: fncs[row](x, u_, **kwargs),
                                   **kwargs).T
    return B
